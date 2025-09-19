import argparse
import os

import numpy as np
import torch
import wandb
from alive_progress import alive_bar
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoConfig

from utils.criterion import FocalLoss
from utils.dataset import DnaPassportDelta
from utils.model import FineTuneModel

wandb.login(force=True)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data Augmentation Transforms
    train_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='train',
                                     anomaly_detection=False)
    test_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='test',
                                    anomaly_detection=False)
    val_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='val',
                                   anomaly_detection=False)
    # [952, 518264]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # configuration
    config = AutoConfig.from_pretrained('configs/dnabert2.json')
    model = FineTuneModel(config).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, capturable=True)
    global_steps = args.epoch_count * len(train_loader)
    warmup_percentage = 0.1
    # make linear warmup schedule and linear decay schedule
    warmup_steps = int(global_steps * warmup_percentage)
    lr_lambda = lambda step: min(1.0, step / warmup_steps) if step <= warmup_steps else max(0.0, 1.0 - (
                step - warmup_steps) / (global_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    criterion = FocalLoss()

    # right after you build `model` (and before training starts)
    wandb.init(project=getattr(args, "wandb_project", "unknown"),
               name=getattr(args, "run_name", f"run_seed{args.seed}"),
               config={"lr": args.learning_rate, "epochs": args.epoch_count, "batch_size": args.batch_size,
                       "seed": args.seed, "model_cfg": "configs/dnabert2.json",
                       "loss": "FocalLoss(alpha=0.95,gamma=2.0)"})
    # optional but useful (limit freq to reduce overhead)
    wandb.watch(model, log="gradients", log_freq=200)
    '''Initial Evaluation'''
    best_auc = float("-inf")
    last_epoch_loss = -1
    last_auc = float("nan")
    last_val_loss = float("nan")
    step = 0
    for epoch in range(1, args.epoch_count + 1):

        epoch_loss = 0
        model.train()

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
                logits = model(**batch_data)
                loss = criterion(logits, labels)
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

        model.eval()
        val_losses, probs, targs = [], [], []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in
                              batch_data.items()}
                labels = batch_data.pop('labels')  # float [B,1]
                logits = model(**batch_data)  # [B,1]
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                p = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
                y = labels.view(-1).detach().cpu().numpy()
                probs.append(p)
                targs.append(y)
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        probs = np.concatenate(probs) if probs else np.array([])
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
        pred = (probs >= 0.5).astype(np.int32)
        acc = accuracy_score(targs, pred) if targs.size else float("nan")

        last_acc = acc
        last_auc = auc
        last_val_loss = val_loss

        # wandb logging
        wandb.log({"val/loss": val_loss, "val/aurochs": auc, "val/auprc": auprc, "val/acc": acc, "epoch": epoch})

        if last_auc > best_auc:
            best_auc = last_auc

            checkpoint = {'args': args, 'model_config': config, 'best_acc': best_auc,
                          'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
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
    parser.add_argument('--wandb_project', type=str, default='cell-passport-focal',
                        help='Weights & Biases project name')

    parser.add_argument('--epoch_count', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--eval_routine', type=int, default=10, help='Log routine')
    parser.add_argument("--saving_path", type=str, default="outputs/focal/", help="Path to save the model")
    # Parse arguments and return the config object
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    config_ = get_config()
    run(config_)
