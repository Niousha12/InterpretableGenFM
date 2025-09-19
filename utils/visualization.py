import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


def plot_dna_token_scores_line(ref_tokens, ref_scores, alt_tokens=None, alt_scores=None, ref_mask=None, alt_mask=None,
        title="Token scores (line plot)", show_delta=True, delta_label_mode="pair",
        # 'ref' | 'alt' | 'pair' | None  (Δ row x-tick labels)
        tick_rotation=45, tick_fontsize=11, max_tick_labels=40, figure_scale_per_token=0.32, min_fig_width=10,
        row_height=2.2, pad_tokens=("[PAD]", "<pad>", "PAD"), save_path=None, ):
    """Line/scatter plot of token scores with optional Δ and pair labels."""

    # --- helpers ---
    def _mask(tokens, scores, mask):
        tokens = np.array(tokens, dtype=object)
        scores = np.asarray(scores, dtype=float)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
        else:
            mask = np.ones(len(tokens), dtype=bool)
        pad_like = np.array([t in pad_tokens for t in tokens], dtype=bool)
        keep = mask & ~pad_like
        return tokens[keep].tolist(), scores[keep]

    def _autostep(n, max_labels):
        return max(1, int(np.ceil(n / max_labels)))

    # --- preprocess ---
    ref_tokens, ref_scores = _mask(ref_tokens, ref_scores, ref_mask)
    n = len(ref_tokens)

    have_alt = alt_tokens is not None and alt_scores is not None
    if have_alt:
        alt_tokens, alt_scores = _mask(alt_tokens, alt_scores, alt_mask)
        m = min(len(alt_tokens), n)
        ref_tokens, ref_scores = ref_tokens[:m], ref_scores[:m]
        alt_tokens, alt_scores = alt_tokens[:m], alt_scores[:m]
        n = m

    rows = 1 + int(show_delta and have_alt)

    # --- figure setup (constrained layout to avoid clipping) ---
    fig_w = max(min_fig_width, n * figure_scale_per_token)
    fig_h = rows * row_height
    try:
        fig, axes = plt.subplots(rows, 1, layout="constrained", figsize=(fig_w, fig_h), sharex=True)
    except TypeError:
        fig, axes = plt.subplots(rows, 1, constrained_layout=True, figsize=(fig_w, fig_h), sharex=True)

    if rows == 1:
        axes = [axes]

    x = np.arange(n)
    step = _autostep(n, max_tick_labels)

    # --- REF + ALT ---
    ax0 = axes[0]
    ax0.plot(x, ref_scores, marker="o", linewidth=1.5, label="REF")
    if have_alt:
        ax0.plot(x, alt_scores, marker="s", linewidth=1.5, label="ALT")
    ax0.set_ylabel("Score")
    ax0.legend(loc="best")
    ax0.set_title(title)
    ax0.set_xticks(x[::step])
    ax0.set_xticklabels([ref_tokens[i] for i in range(0, n, step)], rotation=tick_rotation, ha="right",
                        fontsize=tick_fontsize)
    ax0.grid(True, axis="y", linestyle=":", alpha=0.4)

    # --- Δ row with label mode ---
    if show_delta and have_alt:
        delta = np.asarray(alt_scores) - np.asarray(ref_scores)
        ax1 = axes[1]
        ax1.plot(x, delta, marker="d", linewidth=1.5, label="Δ (ALT−REF)")
        ax1.axhline(0, color="black", lw=1, ls="--")
        ax1.set_ylabel("Δ")
        ax1.legend(loc="best")
        # choose labels for Δ row
        if delta_label_mode == "pair":
            labels = [f"{r}→{a}" for r, a in zip(ref_tokens, alt_tokens)]
        elif delta_label_mode == "alt":
            labels = alt_tokens
        elif delta_label_mode == "ref":
            labels = ref_tokens
        else:
            labels = ["" for _ in range(n)]
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=tick_rotation, ha="right",
                            fontsize=tick_fontsize)
        ax1.grid(True, axis="y", linestyle=":", alpha=0.4)
    if save_path is not None:
        plt.savefig(f"{save_path}/{title}.png", dpi=600, bbox_inches='tight')
    else:
        plt.show()


def visualize_dna_token_scores(ref_tokens, ref_scores, alt_tokens=None, alt_scores=None, ref_mask=None, alt_mask=None,
        title="Per-token scores (REF vs ALT)", show_delta=True, cmap_name="viridis", token_render="ticks",
        # 'ticks' | 'inside' | 'none'
        delta_label_mode="pair",  # 'ref' | 'alt' | 'pair' | None  (for Δ row when token_render='ticks')
        tick_rotation=45, tick_fontsize=11, token_fontsize_inside=9, max_tick_labels=40, figure_scale_per_token=0.32,
        # width (inches) per token
        min_fig_width=10, row_height=1.9, extra_bottom_pad=0.0,  # e.g., 0.04–0.12 if Δ labels are very tall
        pad_tokens=("[PAD]", "<pad>", "PAD"), save_path=None, ):
    """
    Visualize per-token scores for REF and (optionally) ALT sequences,
    plus a delta row (ALT - REF). Supports REF/ALT token labels and pair labels.

    Parameters
    ----------
    ref_tokens, alt_tokens : list[str]
    ref_scores, alt_scores : 1D array-like[float]
    *_mask : 1D {0,1} mask to keep tokens (pads dropped automatically)
    token_render : 'ticks' (labels under bars), 'inside' (labels inside bars), or 'none'
    delta_label_mode : which labels to put under Δ row ('ref', 'alt', 'pair', or None)
    """

    # ---------- helpers ----------
    def _mask(tokens, scores, mask):
        tokens = np.array(tokens, dtype=object)
        scores = np.asarray(scores, dtype=float)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
        else:
            mask = np.ones(len(tokens), dtype=bool)
        pad_like = np.array([t in pad_tokens for t in tokens], dtype=bool)
        keep = mask & ~pad_like
        return tokens[keep].tolist(), scores[keep]

    def _autostep(n, max_labels):
        return max(1, int(np.ceil(n / max_labels)))

    def _text_color_for(value, vmin, vmax):
        t = 0 if vmax == vmin else (value - vmin) / (vmax - vmin)
        return "white" if t > 0.55 else "black"

    # ---------- preprocess ----------
    ref_tokens, ref_scores = _mask(ref_tokens, ref_scores, ref_mask)
    n = len(ref_tokens)

    have_alt = alt_tokens is not None and alt_scores is not None
    if have_alt:
        alt_tokens, alt_scores = _mask(alt_tokens, alt_scores, alt_mask)
        m = min(len(alt_tokens), n)
        ref_tokens, ref_scores = ref_tokens[:m], ref_scores[:m]
        alt_tokens, alt_scores = alt_tokens[:m], alt_scores[:m]
        n = m

    rows = 1 + int(have_alt) + int(have_alt and show_delta)

    # ---------- figure sizing & layout (choose constrained layout from the start) ----------
    fig_w = max(min_fig_width, n * figure_scale_per_token)
    fig_h = rows * row_height
    fig_kw = dict(figsize=(fig_w, fig_h), sharex=True, gridspec_kw=dict(hspace=0.35))

    try:
        # Matplotlib ≥ 3.8
        fig, axes = plt.subplots(rows, 1, layout="constrained", **fig_kw)
    except TypeError:
        # Older Matplotlib
        fig, axes = plt.subplots(rows, 1, constrained_layout=True, **fig_kw)

    if rows == 1:
        axes = [axes]

    # ---------- color scaling ----------
    if have_alt:
        allvals = np.concatenate([np.asarray(ref_scores), np.asarray(alt_scores)])
        vmin, vmax = float(allvals.min()), float(allvals.max())
    else:
        vmin, vmax = float(np.min(ref_scores)), float(np.max(ref_scores))
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    x = np.arange(n)
    bar_height = 1.0

    # ---------- draw a single row ----------
    def _draw_row(ax, tokens, scores, label):
        ax.bar(x, np.full_like(scores, bar_height), bottom=0, width=1.0, color=cmap(norm(scores)), edgecolor="none")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=12)

        if token_render == "inside":
            for i, (tok, val) in enumerate(zip(tokens, scores)):
                ax.text(i, bar_height * 0.5, tok, ha="center", va="center", fontsize=token_fontsize_inside, rotation=90,
                        color=_text_color_for(val, vmin, vmax), clip_on=True)
            ax.set_xticks([])
        elif token_render == "ticks":
            step = _autostep(n, max_tick_labels)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([tokens[i] for i in range(0, n, step)], rotation=tick_rotation, ha="right",
                               fontsize=tick_fontsize)
            ax.grid(axis="x", linestyle=":", alpha=0.2)
        else:
            ax.set_xticks([])

    # ---------- draw rows ----------
    _draw_row(axes[0], ref_tokens, ref_scores, "REF")

    if have_alt:
        _draw_row(axes[1], alt_tokens, alt_scores, "ALT")

        if show_delta:
            delta = np.asarray(alt_scores) - np.asarray(ref_scores)
            dmax = max(abs(delta.min()), abs(delta.max()), 1e-12)
            dnorm = Normalize(vmin=-dmax, vmax=dmax)
            d_cmap = get_cmap("coolwarm")
            ax = axes[2]
            ax.bar(x, np.full_like(delta, bar_height), width=1.0, color=d_cmap(dnorm(delta)), edgecolor="none")
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_yticks([])
            ax.set_ylabel("Δ", rotation=0, ha="right", va="center", fontsize=12)

            if token_render == "ticks" and delta_label_mode:
                step = _autostep(n, max_tick_labels)
                if delta_label_mode == "pair":
                    labels = [f"{r}→{a}" for r, a in zip(ref_tokens, alt_tokens)]
                elif delta_label_mode == "alt":
                    labels = alt_tokens
                elif delta_label_mode == "ref":
                    labels = ref_tokens
                else:
                    labels = [""] * n
                ax.set_xticks(x[::step])
                ax.set_xticklabels([labels[i] for i in range(0, n, step)], rotation=tick_rotation, ha="right",
                                   fontsize=tick_fontsize)
                ax.grid(axis="x", linestyle=":", alpha=0.2)
            else:
                ax.set_xticks([])

    # ---------- shared colorbar ON TOP (add after axes exist) ----------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    try:
        cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", location="top", pad=0.08, fraction=0.06)
    except TypeError:
        # Older Matplotlib: put it near the top; adjust padding below if needed
        cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", pad=0.25, fraction=0.06)
    cbar.set_label("Score", fontsize=tick_fontsize)

    # title
    fig.suptitle(title, fontsize=tick_fontsize)

    # Optional extra margin if very long tick labels still clip at the bottom
    if extra_bottom_pad > 0:
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            pass
        # Nudge bottom space; works with both engines
        plt.subplots_adjust(bottom=min(0.2, 0.08 + extra_bottom_pad))

    if save_path is not None:
        plt.savefig(f"{save_path}/{title}.png", dpi=600, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
