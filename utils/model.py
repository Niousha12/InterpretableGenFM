import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(nn.Conv1d(channels, channels, kernel_size, padding=padding), nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding), )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_b_l_c):
        x = x_b_l_c.transpose(1, 2)  # (B, C, L)
        out = self.net(x).transpose(1, 2)  # back to (B, L, C)
        out = self.dropout(out)
        return self.norm(x_b_l_c + out)


class ResidualConvStack(nn.Module):
    def __init__(self, channels, num_layers=5, kernel_size=5, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock1D(channels, kernel_size, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FineTuneModel(nn.Module):
    PRETRAINED_MODEL_NAME = "jaandoui/DNABERT2-AttentionExtracted"

    def __init__(self, config):
        super().__init__()

        self.base_model = AutoModel.from_pretrained(self.PRETRAINED_MODEL_NAME, trust_remote_code=True)

        if config.freeze_base_model:
            self._freeze_parameters()

        self.conv_layers = self.conv_layers = ResidualConvStack(768, num_layers=5, kernel_size=5, dropout=0.1)
        if config.classification_head == "non-linear":

            act_cls_name = config.cls_activation.capitalize()
            if not hasattr(self, act_cls_name):
                act_cls_name = act_cls_name.upper()
            try:
                activation = getattr(nn, act_cls_name)()
            except AttributeError:
                raise ValueError(f"Activation '{config.cls_activation}' not found in torch.nn")

            self.classification_head = nn.Sequential(nn.Dropout(config.final_dropout),
                nn.Linear(config.hidden_size, config.hidden_size), activation, nn.Dropout(config.final_dropout),
                nn.Linear(config.hidden_size, config.num_labels))
        elif config.classification_head == "linear":
            self.classification_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.num_labels))  # self.gradient_checkpointing = False

    def _freeze_parameters(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        self._requires_grad = False

    @torch.no_grad()
    def _safe_length(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return per-example valid lengths (B,) as float, clamped to avoid div-by-zero."""
        lengths = attention_mask.sum(dim=1) if attention_mask is not None else None
        if lengths is None:
            return None
        return lengths.clamp(min=1).to(torch.float32)

    def forward(self, ref, alt, ref_att, alt_att, **hf_kwargs):
        # 0. Move the inputs to the same device as the model
        device = next(self.parameters()).device
        ref = ref.to(device)
        alt = alt.to(device)
        ref_att = ref_att.to(device) if ref_att is not None else None
        alt_att = alt_att.to(device) if alt_att is not None else None

        # 1. Base model forward → last hidden state (B, C, L, 768)

        base_out_ref = self.base_model(input_ids=ref, attention_mask=ref_att, **hf_kwargs)
        ref_attentions_out = torch.stack(base_out_ref[3], dim=1)
        base_out_alt = self.base_model(input_ids=alt, attention_mask=alt_att, **hf_kwargs)
        alt_attentions_out = torch.stack(base_out_alt[3], dim=1)

        attention_mask = ref_att * alt_att

        delta = base_out_ref[0] - base_out_alt[0]
        x = delta.squeeze(1)

        # 2. Residual Conv stack (expects B, L, C)
        x = self.conv_layers(x)  # (B, L, 768)

        # 3. Mask-aware mean pooling over sequence length
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)  # (B, L, 1)
            x_sum = (x * mask).sum(dim=1)  # (B, 768)
            lengths = self._safe_length(attention_mask).to(x.dtype).unsqueeze(-1)  # (B, 1)
            pooled = x_sum / lengths  # (B, 768)
        else:
            pooled = x.mean(dim=1)  # (B, 768), unmasked fallback

        # 4. Final Projection Layer
        pooled = self.classification_head(pooled)  # (B, num_labels)

        return {"logits": pooled, "ref_attentions": ref_attentions_out, "alt_attentions": alt_attentions_out}


if __name__ == "__main__":
    config_ = AutoConfig.from_pretrained('configs/dnabert2.json')
    model = FineTuneModel(config_).cuda()

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    ref_seq = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    alt_seq = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGT"

    tok_ref = tokenizer(ref_seq, return_tensors='pt')
    tok_alt = tokenizer(alt_seq, return_tensors='pt')

    # Forward pass through the model
    output = model(ref=tok_alt['input_ids'].cuda(), ref_att=tok_alt['attention_mask'].cuda(),
                   alt=tok_ref['input_ids'].cuda(), alt_att=tok_ref['attention_mask'].cuda(), output_attentions=True,
                   return_dict=True)
    print(output.shape)
