"""Explain models fitted on all races: SHAP (gradient boosting) and integrated gradients (neural).

python experiments/run_explain.py [--encoder tcn]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from race_predictor.evaluation.explain import (  # noqa: E402
    attribution_by_channel,
    attribution_by_period,
    integrated_gradients,
    shap_tabular,
    static_feature_names,
)
from race_predictor.evaluation.runner import load_dataset  # noqa: E402
from race_predictor.models import HistGradientBoostingBaseline  # noqa: E402
from race_predictor.models.sequence import SequenceModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--encoder", default="tcn", choices=["gru", "tcn", "transformer"])
    ap.add_argument("--figures", type=Path, default=Path("docs/figures"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_EXPLAINABILITY.md"))
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.data)

    # 1. SHAP on the tabular model
    gb = HistGradientBoostingBaseline().fit(ds)
    shap_df = shap_tabular(gb, ds)
    shap_imp = shap_df.abs().mean().sort_values(ascending=False)
    ax = shap_imp.head(12)[::-1].plot.barh(color="#4c78a8", figsize=(7, 4.5))
    ax.set_xlabel("mean |SHAP| (s/km of predicted pace)")
    ax.set_title("Gradient boosting: feature importance (SHAP)")
    plt.tight_layout()
    plt.savefig(args.figures / "shap_importance.png", dpi=150)
    plt.close()

    # 2. Integrated gradients on a sequence model
    model = SequenceModel(args.encoder).fit(ds)
    seq_attr, static_attr = integrated_gradients(model, ds)
    periods = attribution_by_period(ds, seq_attr)
    channels = attribution_by_channel(seq_attr)
    static_imp = dict(zip(static_feature_names(), abs(static_attr).mean(axis=0), strict=True))
    static_imp = dict(sorted(static_imp.items(), key=lambda kv: -kv[1]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(len(periods)), periods["mean_abs_net_s_km"], color="#4c78a8")
    axes[0].set_xticks(range(len(periods)), [p.split("-")[0] for p in periods["period"]])
    axes[0].set(
        xlabel="days before race (start of 28-day period)",
        ylabel="mean |net IG| (s/km)",
        title=f"{model.name}: which training period matters",
    )
    channels[::-1].plot.barh(ax=axes[1], color="#f58518")
    axes[1].set(xlabel="mean |net IG| (s/km)", title="which run feature matters")
    plt.tight_layout()
    plt.savefig(args.figures / "integrated_gradients.png", dpi=150)
    plt.close()

    args.out.write_text(
        "# Explainability\n\nModels are fitted on all races (explaining, not evaluating). "
        "Attributions are in s/km of predicted race pace.\n\n"
        "## SHAP - gradient boosting (tabular features)\n\n"
        + shap_imp.head(12).round(2).rename("mean |SHAP| s/km").to_markdown()
        + "\n\n"
        "![SHAP](figures/shap_importance.png)\n\n"
        f"## Integrated gradients - {model.name}\n\n"
        "Net attribution by 28-day period before the race (signed within a race, then |.| averaged):\n\n"
        + periods.round(2).to_markdown(index=False)
        + "\n\n"
        "By run feature:\n\n"
        + channels.round(2).rename("mean |net IG| s/km").to_markdown()
        + "\n\n"
        "Static features:\n\n"
        + "\n".join(f"- `{k}`: {v:.2f}" for k, v in list(static_imp.items())[:8])
        + "\n\n"
        "![IG](figures/integrated_gradients.png)\n"
    )
    print(args.out.read_text())


if __name__ == "__main__":
    main()
