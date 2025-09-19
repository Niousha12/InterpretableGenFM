import argparse
import os

import numpy as np
import torch.nn as nn
import wandb
from alive_progress import alive_bar
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from transformers import AutoModel, BertModel, BertConfig

from utils.dataset import DnaPassportDelta
from utils.criterion import kd_loss_from_states
from utils.helpers import IdentityLayer, pad_input

wandb.login(force=True)

import torch


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data Augmentation Transforms
    train_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='train',
                                     anomaly_detection=True)
    test_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='test',
                                    anomaly_detection=False)
    val_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='val',
                                   anomaly_detection=False)
    # [952, 518264]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # define models
    teacher = AutoModel.from_pretrained("jaandoui/DNABERT2-AttentionExtracted", output_hidden_states=True,
                                        output_attentions=True, trust_remote_code=True).cuda()
    teacher.pooler = IdentityLayer()
    tconf = teacher.config
    sconf = BertConfig(vocab_size=tconf.vocab_size,  # 4096
                       hidden_size=tconf.hidden_size,  # 768
                       use_position_embeddings=False, num_hidden_layers=6,  # distilled depth
                       num_attention_heads=tconf.num_attention_heads,  # 12
                       intermediate_size=tconf.intermediate_size,
                       # keep teacher’s (GLU effective mapping handled inside model)
                       hidden_act=tconf.hidden_act,  # "gelu"
                       max_position_embeddings=tconf.max_position_embeddings, type_vocab_size=tconf.type_vocab_size,
                       # 2
                       layer_norm_eps=tconf.layer_norm_eps, output_attentions=True, output_hidden_states=True,
                       return_dict=True,
                       classifier_dropout=tconf.classifier_dropout if hasattr(tconf, "classifier_dropout") else 0.1,
                       problem_type=teacher.config.problem_type if hasattr(teacher.config, "problem_type") else None, )
    student = BertModel(sconf).cuda()

    student.embeddings.position_embeddings = nn.Embedding(student.config.max_position_embeddings,
                                                          student.config.hidden_size)
    # initialize weights to zero and freeze
    nn.init.zeros_(student.embeddings.position_embeddings.weight)
    student.embeddings.position_embeddings.weight.requires_grad = False

    student = student.cuda()
    teacher = teacher.cuda()

    # Define The Layer Mapping
    layer_map = {2: 1, 4: 2, 7: 3, 9: 4, 11: 5}

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, capturable=True)
    global_steps = args.epoch_count * len(train_loader)
    warmup_percentage = 0.1
    # make linear warmup schedule and linear decay schedule
    warmup_steps = int(global_steps * warmup_percentage)
    lr_lambda = lambda step: min(1.0, step / warmup_steps) if step <= warmup_steps else max(0.0, 1.0 - (
                step - warmup_steps) / (global_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = MSELoss()

    # right after you build `model` (and before training starts)
    wandb.init(project=getattr(args, "wandb_project", "cell-passport_500k_kd"),
               name=getattr(args, "run_name", f"run_seed{args.seed}"),
               config={"lr": args.learning_rate, "epochs": args.epoch_count, "batch_size": args.batch_size,
                       "seed": args.seed, "model_cfg": "configs/dnabert2.json",
                       "loss": "FocalLoss(alpha=0.95,gamma=2.0)"})
    # optional but useful (limit freq to reduce overhead)
    wandb.watch(student, log="gradients", log_freq=200)
    '''Initial Evaluation'''
    best_auc = float("-inf")
    last_epoch_loss = -1
    last_auc = float("nan")
    last_val_loss = float("nan")
    step = 0
    for epoch in range(1, args.epoch_count + 1):

        epoch_loss = 0
        teacher.eval()
        student.train()

        total = len(train_loader)

        with alive_bar(total, title=f"Epoch {epoch}", dual_line=False, force_tty=True, length=10, enrich_print=True,
                       max_cols=200) as bar:
            for batch_idx, batch_data in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)

                # move batch to CUDA
                batch_data = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in
                              batch_data.items()}
                labels = batch_data.pop('labels')

                # forward/backward
                # Prepare Inputs for the teacher and student models
                input_ids = torch.cat((batch_data['ref'].unsqueeze(1), batch_data['alt'].unsqueeze(1)),
                                      dim=1)  # [B, 2, L]
                attention_mask = torch.cat((batch_data['ref_att'].unsqueeze(1), batch_data['alt_att'].unsqueeze(1)),
                                           dim=1)  # [B, 2, L]
                B, C, L = input_ids.shape
                input_ids = input_ids.reshape(B * C, L)  # [B*C, L]
                attention_mask = attention_mask.reshape(B * C, L)  # [B*C, L]
                with torch.no_grad():
                    teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                              output_all_encoded_layers=True)
                    teacher_hidden_states = teacher_outputs[0]

                student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)

                student_hidden_states = student_outputs["hidden_states"]

                # Compute loss (e.g., MSE between teacher and student hidden states)
                loss = kd_loss_from_states(teacher_hidden_states, student_hidden_states, attention_mask, B, C, L,
                                           layer_map, criterion)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

                step += 1

                wandb.log({"train/loss_step": loss.item(), "train/lr": optimizer.param_groups[0]["lr"], "epoch": epoch,
                           "step": step})
                bar.text(
                    f"loss={loss.item():.4f} last_epoch_loss={last_epoch_loss:.4f} last_val_loss={last_val_loss:.4f} auc={last_auc:.4f}, best_auc={best_auc:.4f}")
                bar()

        last_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({"train/epoch_loss": last_epoch_loss, "epoch": epoch})

        # Evaluation

        student.eval()
        val_losses, probs, targs = [], [], []

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in
                              batch_data.items()}
                labels = batch_data.pop('labels')  # float [B,1]

                # prepare Inputs for the student model
                input_ids = torch.cat((batch_data['ref'].unsqueeze(1), batch_data['alt'].unsqueeze(1)), dim=1)
                attention_mask = torch.cat((batch_data['ref_att'].unsqueeze(1), batch_data['alt_att'].unsqueeze(1)),
                                           dim=1)
                B, C, L = input_ids.shape
                input_ids = input_ids.reshape(B * C, L)
                attention_mask = attention_mask.reshape(B * C, L)

                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=True)
                teacher_hidden_states = teacher_outputs[0]

                student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                student_hidden_states = student_outputs["hidden_states"]

                loss, logits = kd_loss_from_states(teacher_hidden_states, student_hidden_states,
                                                    attention_mask, B, C, L, layer_map, criterion,
                                                    enable_logits=True)
                val_losses.append(loss.item())
                anomaly_score = logits.detach().cpu().numpy()
                y = labels.view(-1).detach().cpu().numpy()
                probs.append(anomaly_score)
                targs.append(y)
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        probs = np.concatenate(probs) if probs else np.array([])
        print(probs)
        print(targs)
        print(np.sum(targs))
        targs = np.concatenate(targs) if targs else np.array([])

        # metrics (guard against single-class edge cases)
        try:
            auc = roc_auc_score(targs, probs)
        except Exception as e:
            auc = float("nan")
        try:
            auprc = average_precision_score(targs, probs)
        except Exception as e:
            auprc = float("nan")
        # thresholded accuracy (0.5)

        last_auc = auc
        last_val_loss = val_loss

        # wandb logging
        wandb.log({"val/loss": val_loss, "val/aurochs": auc, "val/auprc": auprc, "epoch": epoch})

        if last_auc > best_auc:
            best_auc = last_auc

            checkpoint = {'args': args, 'model_config': sconf, 'best_acc': best_auc,
                          'model_state_dict': student.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch,
                          # if you want to store the current epoch
                          }
            ckpt_path = os.path.join(args.saving_path, f"best_model_epoch{epoch}.pth")
            torch.save(checkpoint, ckpt_path)
            wandb.log({"val/best_auroc": best_auc, "checkpoint_path": ckpt_path, "epoch": epoch})


def get_config():
    parser = argparse.ArgumentParser(description="Configuration for the script")

    # Add arguments with default values
    parser.add_argument('--seed', type=int, default=123, help='Seed for reproducibility')
    parser.add_argument('--wandb_project', type=str, default='cell-passport-kd', help='WandB project name')

    parser.add_argument('--epoch_count', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--eval_routine', type=int, default=10, help='Log routine')
    parser.add_argument("--saving_path", type=str, default="outputs/kd/", help="Path to save the model")
    # Parse arguments and return the config object
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    config_ = get_config()
    run(config_)
