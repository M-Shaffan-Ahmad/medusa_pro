#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


MODE_LABELS = {
    "medusa_base": "Medusa base",
    "turboquant_prod_b4_full_tree": "TurboQuant b4",
    "turboquant_prod_outlier_k4v4_c16_full_tree": "TurboQuant outlier",
}

MODE_COLORS = {
    "medusa_base": "#2563eb",
    "turboquant_prod_b4_full_tree": "#dc2626",
    "turboquant_prod_outlier_k4v4_c16_full_tree": "#7c3aed",
}


def parse_context(category):
    match = re.search(r"(\d+)t$", str(category))
    if not match:
        raise ValueError(f"Cannot parse context from category={category!r}")
    return int(match.group(1))


def parse_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def fmt_value(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def load_rows(paths):
    rows = []
    seen = set()
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["context"] = parse_context(row["category"])
                for key in (
                    "tokens",
                    "total_s",
                    "tps",
                    "speedup_vs_base",
                    "peak_alloc_mb",
                    "fp16_kv_mb_est",
                    "turbo_vq_kv_mb_est",
                    "turbo_vq_transfer_reduction",
                    "prefix_match_vs_base",
                ):
                    row[key] = parse_float(row.get(key))
                unique = (row["context"], row["mode"])
                if unique in seen:
                    continue
                seen.add(unique)
                rows.append(row)
    return sorted(rows, key=lambda item: (item["context"], item["mode"]))


def rows_for(rows, mode):
    return [row for row in rows if row["mode"] == mode]


def save_line_plot(rows, y_key, ylabel, title, path, modes=None, y_floor_zero=True):
    if modes is None:
        modes = sorted({row["mode"] for row in rows})
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for mode in modes:
        data = rows_for(rows, mode)
        if not data:
            continue
        data = [row for row in data if not math.isnan(row[y_key])]
        if not data:
            continue
        ax.plot(
            [row["context"] for row in data],
            [row[y_key] for row in data],
            marker="o",
            linewidth=2.4,
            color=MODE_COLORS.get(mode),
            label=MODE_LABELS.get(mode, mode),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["context"] for row in rows}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1024)}k"))
    ax.set_xlabel("Prompt context tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.28)
    annotate_missing_base(ax, rows, y_key)
    if y_floor_zero:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def missing_base_contexts(rows):
    base_contexts = {row["context"] for row in rows_for(rows, "medusa_base")}
    turbo_contexts = {row["context"] for row in rows_for(rows, "turboquant_prod_b4_full_tree")}
    return sorted(turbo_contexts - base_contexts)


def annotate_missing_base(ax, rows, y_key):
    missing = missing_base_contexts(rows)
    if not missing:
        return
    ymin, ymax = ax.get_ylim()
    y = ymax * 0.92 if ymax > 0 else 1.0
    for ctx in missing:
        ax.scatter(
            [ctx],
            [y],
            marker="x",
            s=80,
            linewidths=2,
            color=MODE_COLORS["medusa_base"],
            zorder=5,
        )
        ax.annotate(
            "base OOM",
            (ctx, y),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=9,
            color=MODE_COLORS["medusa_base"],
        )


def fit_line(xs, ys):
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = (n * sxx) - (sx * sx)
    if denom == 0:
        return 0.0, sy / max(1, n)
    slope = ((n * sxy) - (sx * sy)) / denom
    intercept = (sy - (slope * sx)) / n
    return slope, intercept


def estimate_large_context(rows, contexts):
    base = rows_for(rows, "medusa_base")
    turbo = rows_for(rows, "turboquant_prod_b4_full_tree")
    # Use the larger measured points for trend estimates. They are less affected
    # by fixed model/warmup overhead than the 1k/2k points.
    base_fit = [row for row in base if row["context"] >= 4096]
    turbo_fit = [row for row in turbo if row["context"] >= 8192]
    base_slope, base_intercept = fit_line(
        [row["context"] for row in base_fit],
        [row["total_s"] for row in base_fit],
    )
    turbo_slope, turbo_intercept = fit_line(
        [row["context"] for row in turbo_fit],
        [row["total_s"] for row in turbo_fit],
    )

    observed_turbo = {row["context"]: row for row in turbo}
    estimates = []
    for ctx in contexts:
        base_time = max(0.001, base_intercept + (base_slope * ctx))
        turbo_time = max(0.001, turbo_intercept + (turbo_slope * ctx))
        tokens = 32.0
        tps_base = tokens / base_time
        tps_turbo = tokens / turbo_time
        nearest = max(observed_turbo)
        ref = observed_turbo[nearest]
        kv_per_token_fp16 = ref["fp16_kv_mb_est"] / ref["context"]
        kv_per_token_turbo = ref["turbo_vq_kv_mb_est"] / ref["context"]
        estimates.append(
            {
                "context": ctx,
                "base_tps": tps_base,
                "turbo_tps": tps_turbo,
                "turbo_speedup": tps_turbo / tps_base,
                "fp16_kv_mb": kv_per_token_fp16 * ctx,
                "turbo_kv_mb": kv_per_token_turbo * ctx,
                "kv_saved_mb": (kv_per_token_fp16 - kv_per_token_turbo) * ctx,
            }
        )
    return estimates


def save_memory_plot(rows, path):
    modes = ["medusa_base", "turboquant_prod_b4_full_tree"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for mode in modes:
        data = rows_for(rows, mode)
        data = [row for row in data if not math.isnan(row["peak_alloc_mb"])]
        ax.plot(
            [row["context"] for row in data],
            [row["peak_alloc_mb"] for row in data],
            marker="o",
            linewidth=2.4,
            color=MODE_COLORS[mode],
            label=f"{MODE_LABELS[mode]} peak",
        )
    base = rows_for(rows, "medusa_base")
    turbo = rows_for(rows, "turboquant_prod_b4_full_tree")
    ax.bar(
        [row["context"] * 0.94 for row in base],
        [row["fp16_kv_mb_est"] for row in base],
        width=[row["context"] * 0.08 for row in base],
        alpha=0.24,
        color=MODE_COLORS["medusa_base"],
        label="Estimated fp16 KV",
    )
    ax.bar(
        [row["context"] * 1.06 for row in turbo],
        [row["turbo_vq_kv_mb_est"] for row in turbo],
        width=[row["context"] * 0.08 for row in turbo],
        alpha=0.28,
        color=MODE_COLORS["turboquant_prod_b4_full_tree"],
        label="Estimated TurboQuant KV",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["context"] for row in rows}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1024)}k"))
    ax.set_xlabel("Prompt context tokens")
    ax.set_ylabel("CUDA memory, MB")
    ax.set_title("Total Peak Memory vs. Estimated KV Cache Memory")
    ax.grid(True, which="both", linestyle="--", alpha=0.28)
    annotate_missing_base(ax, rows, "peak_alloc_mb")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_expected_plot(estimates, path):
    fig, ax1 = plt.subplots(figsize=(9.5, 5.6))
    x = [item["context"] for item in estimates]
    ax1.plot(
        x,
        [item["turbo_speedup"] for item in estimates],
        marker="o",
        linewidth=2.4,
        color="#ea580c",
        label="Estimated current TurboQuant speed ratio",
    )
    ax1.axhline(1.0, color="#111827", linewidth=1.3, linestyle="--", alpha=0.65)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda value, _: f"{int(value/1024)}k"))
    ax1.set_xlabel("Prompt context tokens")
    ax1.set_ylabel("TurboQuant TPS / base TPS")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, which="both", linestyle="--", alpha=0.28)

    ax2 = ax1.twinx()
    ax2.bar(
        [item["context"] for item in estimates],
        [item["kv_saved_mb"] for item in estimates],
        width=[item["context"] * 0.16 for item in estimates],
        alpha=0.24,
        color="#16a34a",
        label="Estimated KV memory saved",
    )
    ax2.set_ylabel("Estimated KV memory saved, MB")
    ax1.set_title("Larger-Context Trend Estimate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(rows, estimates, out_dir):
    report_path = out_dir / "turbo_context_findings.md"
    base = {row["context"]: row for row in rows_for(rows, "medusa_base")}
    turbo = {row["context"]: row for row in rows_for(rows, "turboquant_prod_b4_full_tree")}
    contexts = sorted(set(base) | set(turbo))
    missing_base = set(missing_base_contexts(rows))

    lines = [
        "# TurboQuant Context Sweep Findings",
        "",
        "Measured locally on the RTX 3060 Laptop GPU with 32 generated tokens per context point.",
        "The 32k baseline was attempted with the same allocator setting as TurboQuant and OOMed during initial prompt prefill.",
        "",
        "## Plots",
        "",
        "![Throughput](throughput_vs_context.png)",
        "",
        "![Speed Ratio](speed_ratio_vs_context.png)",
        "",
        "![Memory](memory_vs_context.png)",
        "",
        "![Larger Context Estimate](larger_context_estimate.png)",
        "",
        "## Key Measurements",
        "",
        "| Context | Base TPS | TurboQuant b4 TPS | Speed Ratio | Base Peak MB | Turbo Peak MB | Peak Saved MB | Est. KV Saved MB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ctx in contexts:
        b = base.get(ctx)
        t = turbo.get(ctx)
        if t is None:
            continue
        base_tps = "OOM" if ctx in missing_base else fmt_value(b["tps"])
        speed_ratio = "N/A" if math.isnan(t["speedup_vs_base"]) else f"{t['speedup_vs_base']:.3f}"
        base_peak = "OOM" if ctx in missing_base else fmt_value(b["peak_alloc_mb"], 1)
        peak_saved = (
            "N/A"
            if ctx in missing_base
            else fmt_value(b["peak_alloc_mb"] - t["peak_alloc_mb"], 1)
        )
        fp16_kv = t["fp16_kv_mb_est"] if ctx in missing_base else b["fp16_kv_mb_est"]
        lines.append(
            f"| {ctx:,} | {base_tps} | {t['tps']:.2f} | "
            f"{speed_ratio} | {base_peak} | {t['peak_alloc_mb']:.1f} | "
            f"{peak_saved} | {fp16_kv - t['turbo_vq_kv_mb_est']:.1f} |"
        )

    lines += [
        "",
        "## Why Speedup Is Not Showing Yet",
        "",
        "- The benchmark reports total generation speed, not isolated KV-cache bandwidth.",
        "- Model weights, logits, Medusa tree verification, CUDA allocator behavior, and temporary attention tensors dominate short and medium contexts.",
        "- The current TurboQuant path stores KV compressed, but the readable reference attention path still decodes K/V ranges into dense tensors before attention.",
        "- TurboQuant also keeps a recent exact hot window for correctness and fast recent-token access, so memory is compressed plus a small fp16/bf16 tail.",
        "- Compression adds encode/decode/QJL overhead. On this implementation, that overhead is larger than the memory-bandwidth saving up to 16k.",
        "- At 32k, the result changes from a speed question to a capacity question: baseline OOMs on this 6 GB GPU, while TurboQuant b4 completes.",
        "",
        "## 32k Capacity Result",
        "",
        "With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, TurboQuant b4 completed the 32k prompt at 2.20 TPS and 5227.5 MB peak allocation. The baseline run OOMed during initial prompt prefill while trying to allocate another 502 MB.",
        "",
        "## Larger-Context Estimate",
        "",
        "This estimate fits a simple linear time-vs-context trend to the measured larger points. It is a rough projection of the current implementation, not a measured result. Because 32k already puts TurboQuant near the 6 GB GPU limit, 64k+ would likely need 8-bit loading, chunked prefill, a smaller tree, or a GPU with more VRAM.",
        "",
        "| Context | Est. Base TPS | Est. Turbo TPS | Est. Speed Ratio | Est. KV Saved MB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in estimates:
        lines.append(
            f"| {item['context']:,} | {item['base_tps']:.2f} | {item['turbo_tps']:.2f} | "
            f"{item['turbo_speedup']:.3f} | {item['kv_saved_mb']:.1f} |"
        )
    lines += [
        "",
        "Expected outcome with the current code: memory savings should grow close to linearly with context, but throughput likely remains below baseline until the dense decode/reference attention path is fused or avoided.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.summaries)
    main_modes = [
        "medusa_base",
        "turboquant_prod_b4_full_tree",
        "turboquant_prod_outlier_k4v4_c16_full_tree",
    ]
    save_line_plot(
        rows,
        "tps",
        "Tokens per second",
        "Throughput Drops With Context; TurboQuant b4 Narrows the Gap",
        args.out_dir / "throughput_vs_context.png",
        modes=main_modes,
    )
    save_line_plot(
        rows,
        "speedup_vs_base",
        "TPS relative to Medusa base",
        "TurboQuant Speed Ratio Improves With Longer Context, But Stays < 1",
        args.out_dir / "speed_ratio_vs_context.png",
        modes=[
            "turboquant_prod_b4_full_tree",
            "turboquant_prod_outlier_k4v4_c16_full_tree",
        ],
    )
    save_memory_plot(rows, args.out_dir / "memory_vs_context.png")
    estimates = estimate_large_context(rows, [65536, 131072])
    save_expected_plot(estimates, args.out_dir / "larger_context_estimate.png")
    report_path = write_report(rows, estimates, args.out_dir)

    print(f"wrote {args.out_dir}")
    print(f"report {report_path}")


if __name__ == "__main__":
    main()
