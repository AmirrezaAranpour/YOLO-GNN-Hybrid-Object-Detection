"""Step-by-step progression view: how each improvement step moves the focus
variant relative to the original baseline (S0).

Every experiment row carries a `step` tag (see --tag on train.py / eval_pool.py /
eval_b0.py). Rows written before the tag existed have no `step` field and are
treated as the baseline S0. This script groups the focus variant (default
SAGE->GAT) by step, reports mean±std across seeds, and shows the delta vs S0 on
the metrics that matter here: overall mAP@[.5:.95], mAP@.5, small-object mAP, and
lambsquarter AP (the diagnostic class). The B0 and Cand+NMS anchors are shown for
each step when re-measured, else their baseline value.

    python src/eval/compare.py                 # focus = SAGE->GAT
    python src/eval/compare.py GAT             # focus = GAT

Writes reports/results/progression.md + progression_<focus>.png (no retraining).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402
from eval.results import load_rows, mean_std  # noqa: E402

DIAG_CLASS = "lambsquarter"
METRICS = ["mAP50_95", "mAP50", "mAP_small"]


def step_of(row):
    """Normalize the step tag; pre-tag rows and 'baseline' both map to S0."""
    s = row.get("step")
    if s in (None, "baseline", "S0", "s0"):
        return "S0"
    return s


def step_order(rows, steps):
    """Order steps by first appearance in the log, S0 always first."""
    first_seen = {}
    for i, r in enumerate(rows):
        s = step_of(r)
        first_seen.setdefault(s, i)
    ordered = sorted(steps, key=lambda s: (s != "S0", first_seen.get(s, 1 << 30)))
    return ordered


def agg_variant_step(rows, variant, step):
    """mean±std for one (variant, step) over its seeds, deduped by seed (last wins)."""
    by_seed = {}
    for r in rows:
        if r.get("eval_method") != "pycoco" or r.get("label_set", "fine") != "fine":
            continue
        if r["variant"] != variant or step_of(r) != step:
            continue
        by_seed[r.get("seed")] = r
    rs = list(by_seed.values())
    if not rs:
        return None
    out = {"n_seeds": len(rs)}
    for m in METRICS:
        out[m] = mean_std([r.get(m) for r in rs])
    out[DIAG_CLASS] = mean_std(
        [r.get("per_class_mAP50_95", {}).get(DIAG_CLASS) for r in rs])
    return out


def anchor_for_step(rows, variant, step):
    """Anchor value for a step; fall back to the S0 anchor if not re-measured."""
    return agg_variant_step(rows, variant, step) or agg_variant_step(rows, variant, "S0")


def pct(ms):
    return None if ms is None or ms[0] is None else ms[0] * 100


def cell(ms):
    if ms is None or ms[0] is None:
        return "-"
    return f"{ms[0]*100:.2f}±{ms[1]*100:.2f}"


def delta(cur, base):
    a, b = pct(cur), pct(base)
    if a is None or b is None:
        return "-"
    return f"{a-b:+.2f}"


def main(focus="SAGE->GAT"):
    rows = load_rows(REPO_ROOT / "logs/experiments.jsonl")
    steps = {step_of(r) for r in rows}
    steps = step_order(rows, steps)

    # focus variant per step
    focus_by_step = {s: agg_variant_step(rows, focus, s) for s in steps}
    focus_steps = [s for s in steps if focus_by_step[s] is not None]
    base = focus_by_step.get("S0")

    # ---- build markdown ----
    header = ["step", "seeds", "mAP50-95", "Δ vs S0", "mAP50", "mAP_small",
              f"AP:{DIAG_CLASS}", "Δlqtr"]
    lines = [f"# Progression — focus variant: `{focus}`", "",
             "All numbers %, mean±std over seeds. Δ is vs the original baseline (S0).",
             "", "## Focus variant", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for s in focus_steps:
        d = focus_by_step[s]
        lines.append("| " + " | ".join([
            s, str(d["n_seeds"]), cell(d["mAP50_95"]), delta(d["mAP50_95"], base["mAP50_95"]),
            cell(d["mAP50"]), cell(d["mAP_small"]),
            cell(d[DIAG_CLASS]), delta(d[DIAG_CLASS], base[DIAG_CLASS]),
        ]) + " |")

    # ---- anchors per step ----
    lines += ["", "## Anchors (per step; falls back to S0 value if not re-measured)", "",
              "| step | B0 mAP50-95 | Cand+NMS mAP50-95 | best-focus vs B0 |",
              "|---|---|---|---|"]
    for s in focus_steps:
        b0 = anchor_for_step(rows, "B0", s)
        cn = anchor_for_step(rows, "Cand+NMS", s)
        f = focus_by_step[s]
        gap = delta(f["mAP50_95"], b0["mAP50_95"]) if b0 else "-"
        lines.append(f"| {s} | {cell(b0['mAP50_95']) if b0 else '-'} | "
                     f"{cell(cn['mAP50_95']) if cn else '-'} | {gap} |")

    out_md = REPO_ROOT / "reports/results/progression.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    # ---- plot ----
    xs = focus_steps
    ys = [pct(focus_by_step[s]["mAP50_95"]) for s in xs]
    es = [focus_by_step[s]["mAP50_95"][1] * 100 for s in xs]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(xs)), 5))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, color="#4477aa", label=focus)
    b0 = anchor_for_step(rows, "B0", "S0")
    cn = anchor_for_step(rows, "Cand+NMS", "S0")
    if b0:
        ax.axhline(pct(b0["mAP50_95"]), ls="--", color="#228833", label="B0 (baseline)")
    if cn:
        ax.axhline(pct(cn["mAP50_95"]), ls=":", color="#ccbb44",
                   label="Cand+NMS (baseline)")
    ax.set_ylabel("mAP@[.5:.95] (%)")
    ax.set_title(f"Progression of {focus} vs baseline anchors")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    for x, y in zip(xs, ys):
        if y is not None:
            ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    plot_path = REPO_ROOT / f"reports/results/progression_{focus.replace('->','_')}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    # ---- console ----
    print(f"focus variant: {focus}")
    print(f"steps: {focus_steps}")
    for s in focus_steps:
        d = focus_by_step[s]
        print(f"  {s:12s} seeds={d['n_seeds']} "
              f"mAP50-95={cell(d['mAP50_95'])} (Δ {delta(d['mAP50_95'], base['mAP50_95'])}) "
              f"mAP_small={cell(d['mAP_small'])} "
              f"AP:{DIAG_CLASS}={cell(d[DIAG_CLASS])}")
    if b0:
        print(f"  anchor B0 (S0)      mAP50-95={cell(b0['mAP50_95'])}")
    if cn:
        print(f"  anchor Cand+NMS(S0) mAP50-95={cell(cn['mAP50_95'])}")
    print(f"wrote -> {out_md} and {plot_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
