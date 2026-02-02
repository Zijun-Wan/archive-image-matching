import matplotlib.pyplot as plt
from pathlib import Path

def _qnums(per_query):
    # extract query numbers for clean x-axis ticks (e.g., "1", "2", ...)
    names = [r["query"] for r in per_query]
    return [int(n.split(".")[0]) if n.endswith(".png") else n for n in names]

def _grouped_bar(ax, x_idx, groups, values_by_group, width=0.2):
    """
    Draw grouped bars.
    - x_idx: array-like positions (1..N)
    - groups: e.g., [1,3,5,10]
    - values_by_group: dict {k -> [v1, v2, ...] aligned with x_idx}
    """
    offsets = []
    g = len(groups)
    start = - (g - 1) * width / 2
    for i in range(g):
        offsets.append(start + i * width)
    for off, k in zip(offsets, groups):
        xs = [xi + off for xi in x_idx]
        ax.bar(xs, values_by_group[k], width=width, label=f"@{k}")

    ax.set_xticks(x_idx)

def plot_benchmark(per_query, macro, side_note, output_dir=None, show=True):
    """
    per_query: list of dicts produced in your loop
    macro: {"MAP":..., "MRR":..., "Precision@K":{k->...}, "Recall@K":{k->...}, "Hits@K":{k->...}}
    side_note: {"inlier_ratio_threshold":..., "avg_max_inlier_ratio_TP":..., "avg_mean_inlier_ratio_TP":..., "fraction_queries_with_confident_TP":...}
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Per-query arrays ----------
    qs = _qnums(per_query)
    x = list(range(1, len(qs) + 1))

    ap_vals  = [r["AP"]  for r in per_query]
    mrr_vals = [r["MRR"] for r in per_query]

    ks = [1, 3, 5, 10]
    p_at_k = {k: [r[f"Precision@{k}"] for r in per_query] for k in ks}
    r_at_k = {k: [r[f"Recall@{k}"]    for r in per_query] for k in ks}
    h_at_k = {k: [r[f"Hit@{k}"]       for r in per_query] for k in ks}

    # ---------- Plot: Per-query AP ----------
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(x, ap_vals)
    ax.set_title("Per-query Average Precision (AP)")
    ax.set_xlabel("Query")
    ax.set_ylabel("AP")
    ax.set_xticks(x)
    ax.set_xticklabels(qs)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "per_query_AP.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # ---------- Plot: Per-query MRR ----------
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(x, mrr_vals)
    ax.set_title("Per-query Mean Reciprocal Rank (MRR)")
    ax.set_xlabel("Query")
    ax.set_ylabel("MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(qs)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "per_query_MRR.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # ---------- Plot: Per-query P@K (grouped) ----------
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    _grouped_bar(ax, x, ks, p_at_k, width=0.18)
    ax.set_title("Per-query Precision@K")
    ax.set_xlabel("Query")
    ax.set_ylabel("Precision")
    ax.set_xticklabels(qs)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "per_query_P_at_K.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # ---------- Plot: Per-query R@K (grouped) ----------
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    _grouped_bar(ax, x, ks, r_at_k, width=0.18)
    ax.set_title("Per-query Recall@K")
    ax.set_xlabel("Query")
    ax.set_ylabel("Recall")
    ax.set_xticklabels(qs)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "per_query_R_at_K.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # ---------- Plot: Per-query Hit@K (grouped, 0/1) ----------
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    _grouped_bar(ax, x, ks, h_at_k, width=0.18)
    ax.set_title("Per-query Hit@K (1 if any TP in top-K)")
    ax.set_xlabel("Query")
    ax.set_ylabel("Hit (0/1)")
    ax.set_xticklabels(qs)
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "per_query_Hit_at_K.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # ---------- Plot: Aggregate (macro) P@K / R@K / Hits@K ----------
    # P@K
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.bar([str(k) for k in ks], [macro["Precision@K"][k] for k in ks])
    ax.set_title("Macro Precision@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "macro_P_at_K.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # R@K
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.bar([str(k) for k in ks], [macro["Recall@K"][k] for k in ks])
    ax.set_title("Macro Recall@K")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "macro_R_at_K.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # Hits@K
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.bar([str(k) for k in ks], [macro["Hits@K"][k] for k in ks])
    ax.set_title("Macro Hits@K (fraction of queries)")
    ax.set_xlabel("K")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "macro_Hits_at_K.png", dpi=160)
    if show: plt.show()
    plt.close(fig)

    # One more compact figure with MAP & MRR as text (optional)
    fig = plt.figure(figsize=(5, 3))
    ax = plt.gca()
    ax.axis("off")
    txt = (f"MAP = {macro['MAP']:.4f}\n"
           f"MRR = {macro['MRR']:.4f}\n"
           f"Inlier thr = {side_note['inlier_ratio_threshold']}\n"
           f"Avg max inlier (TP) ≈ {side_note['avg_max_inlier_ratio_TP']:.4f}\n"
           f"Avg mean inlier (TP) ≈ {side_note['avg_mean_inlier_ratio_TP']:.4f}\n"
           f"Frac queries with TP ≥ thr = {side_note['fraction_queries_with_confident_TP']:.3f}")
    ax.text(0.05, 0.95, txt, va="top", ha="left", fontsize=11)
    fig.tight_layout()
    if output_dir: fig.savefig(output_dir / "summary_text.png", dpi=160)
    if show: plt.show()
    plt.close(fig)
