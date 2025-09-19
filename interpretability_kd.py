import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import BertModel, BertConfig

from utils.criterion import kd_loss_from_states
from utils.dataset import DnaPassportDelta
from utils.helpers import IdentityLayer, teacher_states
from utils.visualization import visualize_dna_token_scores

test_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='test')

tokenizer = test_dataset.tokenizer

test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

# define models
teacher = AutoModel.from_pretrained("jaandoui/DNABERT2-AttentionExtracted", output_hidden_states=True,
    output_attentions=True, trust_remote_code=True).cuda()

teacher.pooler = IdentityLayer()
tconf = teacher.config
sconf = BertConfig(vocab_size=tconf.vocab_size,  # 4096
    hidden_size=tconf.hidden_size,  # 768
    use_position_embeddings=False, num_hidden_layers=6,  # distilled depth
    num_attention_heads=tconf.num_attention_heads,  # 12
    intermediate_size=tconf.intermediate_size,  # keep teacher’s (GLU effective mapping handled inside model)
    hidden_act=tconf.hidden_act,  # "gelu"
    max_position_embeddings=tconf.max_position_embeddings, type_vocab_size=tconf.type_vocab_size,  # 2
    layer_norm_eps=tconf.layer_norm_eps, output_attentions=True, output_hidden_states=True, return_dict=True,
    classifier_dropout=tconf.classifier_dropout if hasattr(tconf, "classifier_dropout") else 0.1,
    problem_type=teacher.config.problem_type if hasattr(teacher.config, "problem_type") else None, )
student = BertModel(sconf).cuda()

student.embeddings.position_embeddings = nn.Embedding(student.config.max_position_embeddings,
    student.config.hidden_size)
# initialize weights to zero and freeze
nn.init.zeros_(student.embeddings.position_embeddings.weight)
student.embeddings.position_embeddings.weight.requires_grad = False

#  load checkpoint for student

ckpt = torch.load("outputs/best_model_epoch4.pth", map_location="cpu", weights_only=False)
student.load_state_dict(ckpt["model_state_dict"])


student = student.cuda()
teacher = teacher.cuda()

layer_map = {2: 1, 4: 2, 7: 3, 9: 4, 11: 5}

# inputs = inputs.cuda()
# inputs.requires_grad = True

criterion = nn.MSELoss()


# similarity_loss = torch.nn.CosineSimilarity()


def saliency_from_kd(student, teacher, batch_data, layer_map, criterion, norm="l2"):
    # Move to device
    device = next(student.parameters()).device
    batch_data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_data.items()}

    # Prepare inputs (same packing as your training)
    input_ids = torch.cat((batch_data['ref'].unsqueeze(1), batch_data['alt'].unsqueeze(1)), dim=1)  # [B, 2, L]
    attention_mask = torch.cat((batch_data['ref_att'].unsqueeze(1), batch_data['alt_att'].unsqueeze(1)), dim=1)
    B, C, L = input_ids.shape
    assert C == 2, "saliency_from_kd assumes ref/alt pair."
    flat_ids = input_ids.reshape(B * C, L)
    flat_mask = attention_mask.reshape(B * C, L)

    # 1) Teacher hidden states (no grad)
    with torch.no_grad():
        teacher_hidden_states = teacher_states(teacher, flat_ids, flat_mask)

    # 2) Student forward using inputs_embeds so we can get grads wrt embeddings
    student.eval()
    # Build token embeddings and enable grads
    word_emb = student.get_input_embeddings()  # nn.Embedding
    inp_emb = word_emb(flat_ids)  # (B*C, L, H)  <-- non-leaf
    inp_emb = inp_emb.detach()  # <-- make leaf
    inp_emb.requires_grad_(True)  # track grad
    inp_emb.retain_grad()  # store .grad after backward

    # Forward student with inputs_embeds (BERT still adds pos/type embeddings internally)
    out = student(inputs_embeds=inp_emb, attention_mask=flat_mask)
    student_hidden_states = out["hidden_states"]  # tuple(len=layers) of (B*C, L, H)

    # 3) KD loss and backward wrt embeddings
    loss = kd_loss_from_states(teacher_hidden_states, student_hidden_states, flat_mask, B, C, L, layer_map, criterion)
    # Enable grad graph for embeddings only
    student.zero_grad(set_to_none=True)
    if inp_emb.grad is not None:
        inp_emb.grad.zero_()
    loss.backward()

    grads = inp_emb.grad.detach()  # (B*C, L, H)

    # 4) Collapse embedding dim -> per-token saliency
    if norm == "l1":
        sal = grads.abs().sum(dim=-1)  # (B*C, L)
    else:  # l2 default
        sal = grads.pow(2).sum(dim=-1).sqrt()  # (B*C, L)

    # Optional grad×input (can be more faithful)
    grad_x_input = (grads * inp_emb.detach()).sum(-1)  # (B*C, L)

    # 5) Unpack back to (B, 2, L) and split ref/alt
    sal = sal.reshape(B, C, L)
    gxi = grad_x_input.reshape(B, C, L)

    # Mask out pads
    mask = attention_mask.bool()  # (B, 2, L)
    sal = sal * mask + (~mask) * 0.0
    gxi = gxi * mask + (~mask) * 0.0

    # Normalize per sequence for visualization (robust min-max)
    def _norm_per_seq(x):
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-9)

    sal_n = _norm_per_seq(sal)
    gxi_n = _norm_per_seq(gxi)

    # Return both vanilla and grad×input; caller can choose
    out_dict = {'ref_input_ids': input_ids[:, 0, :],  # (B, L)
                'ref_attention_mask': attention_mask[:, 0, :],  # (B, L)
                'alt_input_ids': input_ids[:, 1, :],  # (B, L)
                'alt_attention_mask': attention_mask[:, 1, :],  # (B, L)
                "saliency_ref": sal_n[:, 0, :],  # (B, L)
                "saliency_alt": sal_n[:, 1, :],  # (B, L)
                "saliency_delta": (sal_n[:, 0, :] - sal_n[:, 1, :]),  # signed diff
                "gxi_ref": gxi_n[:, 0, :], "gxi_alt": gxi_n[:, 1, :], "gxi_delta": (gxi_n[:, 0, :] - gxi_n[:, 1, :]),
                "loss_value": float(loss.item()), }
    return out_dict


for idx, batch_data in enumerate(test_loader):
    out = saliency_from_kd(student, teacher, batch_data, layer_map, criterion, norm="l2")
    print(out)
    labels = batch_data.pop('labels')  # float [B,1]

    for i in range(len(labels)):
        label = int(labels[i].item())
        print(f"********************************{label}***************************")
        label = "benign" if label == 0 else "pathologic"

        visualize_dna_token_scores(tokenizer.convert_ids_to_tokens(out['ref_input_ids'][i]),
            out["saliency_ref"][i].cpu().numpy().tolist(), tokenizer.convert_ids_to_tokens(out['alt_input_ids'][i]),
            out["saliency_alt"][i].cpu().numpy().tolist(),
            title=f"Token h scores (REF vs ALT) for {label} sample #{idx * 6 + i}", show_delta=True, tick_rotation=90,
            delta_label_mode="pair", save_path="outputs/attention_maps_v2")
