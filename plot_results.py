import matplotlib.pyplot as plt
import numpy as np

# Results data
k_values = [1, 2, 4, 8, 32]

baseline_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
finetuned_vals = [0.0086, 0.0159, 0.0277, 0.0447, 0.0902]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Shaded improvement area (green tint)
ax.fill_between(
    k_values, baseline_vals, finetuned_vals,
    color="#2ecc71", alpha=0.18, zorder=2, label="_nolegend_"
)

# Baseline flat line
ax.plot(
    k_values, [v * 100 for v in baseline_vals],
    color="#b0bec5", linewidth=2.2,
    marker="o", markersize=7, markerfacecolor="#b0bec5",
    linestyle="--", label="Baseline (Qwen3-4B, no adapter)",
    zorder=3,
)

# Fine-tuned curve
ax.plot(
    k_values, [v * 100 for v in finetuned_vals],
    color="#1565c0", linewidth=2.5,
    marker="o", markersize=8, markerfacecolor="#1565c0",
    linestyle="-", label="Fine-tuned (LoRA on Lean Workbook)",
    zorder=4,
)

# Annotate each finetuned data point
offsets = [(1.1, 0.4), (2.2, 0.4), (4.4, 0.3), (9, 0.3), (33, 0.4)]
for (kx, ky_off), k, v in zip(offsets, k_values, finetuned_vals):
    ax.annotate(
        f"{v*100:.2f}%",
        xy=(k, v * 100),
        xytext=(kx, v * 100 + ky_off + 0.25),
        fontsize=8.5, color="#1565c0",
        ha="left",
    )

# Annotate baseline
ax.text(1.05, 0.15, "0.00% (all k)", color="#90a4ae", fontsize=8.5, style="italic")


ax.set_xscale("log", base=2)
ax.set_xticks(k_values)
ax.set_xticklabels([f"pass@{k}" for k in k_values], fontsize=10)
ax.set_yticks(np.arange(0, 11, 2))
ax.set_yticklabels([f"{v}%" for v in range(0, 11, 2)], fontsize=10)

ax.set_ylim(-0.3, 11)
ax.set_xlim(0.85, 50)

ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cfd8dc")
ax.spines["bottom"].set_color("#cfd8dc")

ax.set_title(
    "Qwen3-4B-Instruct: Baseline vs. Fine-tuned on miniF2F-test\n"
    "(244 problems, 32 samples per problem)",
    fontsize=12, fontweight="bold", pad=14, color="#212121",
)
ax.set_xlabel("k", fontsize=11, color="#424242")
ax.set_ylabel("Fraction of problems solved (%)", fontsize=11, color="#424242")

legend = ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9,
                   edgecolor="#e0e0e0")

plt.tight_layout()
out = "lean_prover_ft/results/minif2f_results.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
plt.close()
