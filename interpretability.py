import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoConfig

from utils.dataset import DnaPassportDelta
from utils.model import FineTuneModel
from utils.visualization import visualize_dna_token_scores

test_dataset = DnaPassportDelta(csv_file_path='dataset/cell_passport_sampled_15k.csv', split='test')

test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Rebuild model from config
config = AutoConfig.from_pretrained('configs/dnabert2.json')
model = FineTuneModel(config).cuda()

# Restore model weights
ckpt = torch.load("outputs/focal/best_model_epoch3.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

model.eval()

tokenizer = test_dataset.tokenizer

tokenized_ref = []
tokenized_alt = []
attention_masks_ref = []
attention_masks_alt = []
labels = []
attention_out_ref = []
attention_out_alt = []


def attention_rollout(attn_layers, add_residual=True, discard_cls_to_others=False, kind=None, label=None, i=None,
                      attention_mask=None):
    """
    attn_layers: Tensor (L, S, S) after head-averaging.
    Returns: vector of length S with propagated importance from [CLS] to all tokens.
    """
    L, S, _ = attn_layers.shape
    A = attn_layers

    # ---- NEW: determine valid (non-pad) length and crop ----
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[0]
        S_valid = int(attention_mask.to(dtype=torch.long).sum().item())
        S_valid = max(1, min(S_valid, S))
        A = A[:, :S_valid, :S_valid]
    else:
        S_valid = S  # keep A as-is

    # Optionally zero out attentions from [CLS] to others (sometimes suggested for ViTs)
    if discard_cls_to_others:
        A = A.clone()
        A[:, 0, 1:] = 0.0

    # Normalize each layer row-wise to be stochastic
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-9)

    # Add residual connection as I + A, then renormalize (per rollout paper)
    if add_residual:
        I = torch.eye(S_valid).to(A.device)
        A = (A + I[None, :, :]) / 2.0

    # Cumulative product across layers (matrix multiply)
    rollout = A[0]
    for l in range(1, L):
        rollout = rollout @ A[l]

    # Importance of each token given [CLS] as the source (row 0)
    # plot the whole rollout matrix
    plt.imshow(rollout.detach().cpu().numpy())
    plt.colorbar()
    plt.title(f"Attention Matrix for {kind}")
    plt.tight_layout()
    plt.savefig(f"outputs/attention_maps/attention_rollout_{kind}_{label}_{i}.png", dpi=300)
    plt.close()
    # plt.show()

    cls_to_tokens = rollout[0]  # (S,)
    # Normalize to [0, 1]
    # cls_to_tokens[0] = 0.0  # optionally zero out CLS itself
    scores = (cls_to_tokens - cls_to_tokens.min()) / (cls_to_tokens.max() - cls_to_tokens.min() + 1e-9)
    return scores  # (S,)


def compute_attention_scores(seq, attentions_map, attention_mask, decoder, kind, label, i):
    # A = torch.stack(attentions, dim=0).squeeze(1)  # (L, H, S, S)
    a = attentions_map.mean(dim=1)  # (L, S, S) average heads
    scores = attention_rollout(a, add_residual=True, kind=kind, label=label, i=i, attention_mask=attention_mask)
    tokens = decoder.convert_ids_to_tokens(seq)

    # Use mask to filter out padding tokens
    mask = attention_mask.bool().cpu().numpy()
    tokens = [tok for tok, m in zip(tokens, mask) if m]
    scores = scores[:len(tokens)].detach().cpu().numpy().tolist()

    return tokens, scores

    # for tok, s in zip(tokens, scores.tolist()):  #     print(f"{tok:>12s}  {s:.3f}")


for batch in test_loader:
    with torch.no_grad():
        outputs = model(**batch, output_attentions=True)
        attention_out_ref.append(outputs['ref_attentions'].detach().cpu())  # list of (B, L, H, S, S)
        attention_out_alt.append(outputs['alt_attentions'].detach().cpu())  # list of (B, L, H, S, S)
        tokenized_ref.append(batch['ref'].cpu())
        tokenized_alt.append(batch['alt'].cpu())
        attention_masks_ref.append(batch['ref_att'].cpu())
        attention_masks_alt.append(batch['alt_att'].cpu())
        labels.append(batch['labels'].cpu())

# Stack all batches
attention_out_ref = torch.cat(attention_out_ref, dim=0)  # (N, L, H, S, S)
attention_out_alt = torch.cat(attention_out_alt, dim=0)  # (N, L, H, S, S)
tokenized_ref = torch.cat(tokenized_ref, dim=0)  # (N, S)
tokenized_alt = torch.cat(tokenized_alt, dim=0)  # (N, S)
attention_masks_ref = torch.cat(attention_masks_ref, dim=0)  # (N, S)
attention_masks_alt = torch.cat(attention_masks_alt, dim=0)  # (N, S)
labels = torch.cat(labels, dim=0)  # (N,)

pathologic_indices = torch.where(labels == 1)[0][:3].tolist()
benign_indices = torch.where(labels == 0)[0][:3].tolist()

for i in range(len(test_dataset)):
    if i not in pathologic_indices and i not in benign_indices:
        continue
    enc = {"ref": tokenized_ref[i].cuda(), "ref_att": attention_masks_ref[i].cuda(), "alt": tokenized_alt[i].cuda(),
        "alt_att": attention_masks_alt[i].cuda(), }
    att_out_ref = attention_out_ref[i]
    att_out_alt = attention_out_alt[i]
    label = labels[i]
    print(f"********************************{label}***************************")
    label = "benign" if label == 0 else "pathologic"

    ref_tokens, ref_scores = compute_attention_scores(enc['ref'], att_out_ref, enc['ref_att'], tokenizer, 'REF', label,
                                                      i)
    alt_tokens, alt_scores = compute_attention_scores(enc['alt'], att_out_alt, enc['alt_att'], tokenizer, 'ALT', label,
                                                      i)

    visualize_dna_token_scores(ref_tokens, ref_scores, alt_tokens, alt_scores,
        title=f"Token h scores (REF vs ALT) for {label} sample #{i}", show_delta=True, tick_rotation=90,
        delta_label_mode="pair", save_path="outputs/attention_maps")
