"""Exploratory Data Analysis module for Hit Song Predictor.

Produces publication-grade figures and statistical evaluations for the
modelling dataset (data/processed/songs.parquet) and the full dataset
(data/processed/songs_full.parquet).

Figures generated:
- reports/figures/class_balance_natural.png
- reports/figures/class_balance_modeling.png
- reports/figures/tempo_distribution.png
- reports/figures/valence_distribution.png
- reports/figures/danceability_vs_energy.png
- reports/figures/correlation_heatmap.png
- reports/figures/duration_by_year.png
- reports/figures/feature_boxplots.png

Tables generated:
- reports/tables/target_correlations.csv
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

AUDIO_FEATURES = [
    'tempo', 'energy', 'danceability', 'valence', 'loudness',
    'acousticness', 'instrumentalness', 'speechiness', 'duration_sec'
]


def display_and_save(fig: plt.Figure, save_path: str) -> None:
    """Save figure at 150 dpi and display without blocking CLI."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show(block=False)
    plt.close(fig)


def plot_class_bar(df: pd.DataFrame, title: str, save_path: str) -> plt.Figure:
    """Plot annotated bar chart of charted vs non-charted class distribution."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['charted'].value_counts()
    total = len(df)
    c0, c1 = counts.get(0, 0), counts.get(1, 0)
    bars = ax.bar(['Not Charted (0)', 'Charted (1)'], [c0, c1], color=['#7F8C8D', '#26407A'], width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + max(c0, c1) * 0.02, f"{h:,}\n({(h / total) * 100:.2f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Track Count", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylim(0, max(c0, c1) * 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    display_and_save(fig, save_path)
    return fig


def plot_class_balance_natural(df_full: pd.DataFrame, save_path: str = "reports/figures/class_balance_natural.png") -> plt.Figure:
    """Plot natural class imbalance before downsampling using songs_full.parquet."""
    return plot_class_bar(df_full, "Natural Class Imbalance Before Downsampling", save_path)


def plot_class_balance_modeling(df: pd.DataFrame, save_path: str = "reports/figures/class_balance_modeling.png") -> plt.Figure:
    """Plot class balance in the final downsampled modelling dataset."""
    return plot_class_bar(df, "Class Balance in Final Modelling Dataset", save_path)


def plot_tempo_distribution(df: pd.DataFrame, save_path: str = "reports/figures/tempo_distribution.png") -> plt.Figure:
    """Plot density-normalised tempo distribution comparing charted vs non-charted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df[df['charted'] == 0]['tempo'], label='Not Charted (0)', color='#7F8C8D', fill=True, alpha=0.3, common_norm=False, ax=ax)
    sns.kdeplot(data=df[df['charted'] == 1]['tempo'], label='Charted (1)', color='#26407A', fill=True, alpha=0.4, common_norm=False, ax=ax)
    ax.set_xlabel("Tempo (BPM)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Tempo Distribution: Charted vs Non-Charted Songs", fontsize=12)
    ax.legend(title="Status")
    ax.grid(True, linestyle='--', alpha=0.5)
    display_and_save(fig, save_path)
    return fig


def plot_valence_distribution(df: pd.DataFrame, save_path: str = "reports/figures/valence_distribution.png") -> plt.Figure:
    """Plot density-normalised valence distribution comparing charted vs non-charted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df[df['charted'] == 0]['valence'], label='Not Charted (0)', color='#7F8C8D', fill=True, alpha=0.3, common_norm=False, ax=ax)
    sns.kdeplot(data=df[df['charted'] == 1]['valence'], label='Charted (1)', color='#26407A', fill=True, alpha=0.4, common_norm=False, ax=ax)
    ax.set_xlabel("Valence (Musical Positiveness, 0.0 - 1.0)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Valence Distribution: Charted vs Non-Charted Songs", fontsize=12)
    ax.legend(title="Status")
    ax.grid(True, linestyle='--', alpha=0.5)
    display_and_save(fig, save_path)
    return fig


def plot_danceability_vs_energy(df: pd.DataFrame, save_path: str = "reports/figures/danceability_vs_energy.png") -> plt.Figure:
    """Scatter plot of danceability vs energy by chart status with transparency."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[df['charted'] == 0]['danceability'], df[df['charted'] == 0]['energy'], color='#7F8C8D', alpha=0.35, s=20, label='Not Charted (0)')
    ax.scatter(df[df['charted'] == 1]['danceability'], df[df['charted'] == 1]['energy'], color='#26407A', alpha=0.85, s=25, edgecolors='white', linewidth=0.5, label='Charted (1)')
    ax.set_xlabel("Danceability (0.0 - 1.0)", fontsize=11)
    ax.set_ylabel("Energy (0.0 - 1.0)", fontsize=11)
    ax.set_title("Danceability vs Energy by Chart Status", fontsize=12)
    ax.legend(title="Status")
    ax.grid(True, linestyle='--', alpha=0.5)
    display_and_save(fig, save_path)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = "reports/figures/correlation_heatmap.png") -> plt.Figure:
    """Plot Pearson correlation heatmap across numeric audio features only."""
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[AUDIO_FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax, square=True, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Pearson Correlation Heatmap of Numeric Audio Features", fontsize=12)
    display_and_save(fig, save_path)
    return fig


def plot_duration_by_year(df: pd.DataFrame, save_path: str = "reports/figures/duration_by_year.png") -> plt.Figure:
    """Plot median duration of Billboard-charted songs across chart years with IQR band."""
    charted_df = df[df['charted'] == 1].dropna(subset=['chart_year'])
    stats_df = charted_df.groupby('chart_year')['duration_sec'].agg(
        median='median', q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    ).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stats_df['chart_year'], stats_df['median'], color='#26407A', marker='o', linewidth=2, label='Median Duration')
    ax.fill_between(stats_df['chart_year'], stats_df['q25'], stats_df['q75'], color='#26407A', alpha=0.2, label='IQR (25th - 75th Percentile)')
    ax.set_title("Median Duration of Billboard-Charted Songs by Chart Year", fontsize=12)
    ax.set_xlabel("Billboard Chart Year", fontsize=11)
    ax.set_ylabel("Duration (seconds)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    display_and_save(fig, save_path)
    return fig


def plot_feature_boxplots(df: pd.DataFrame, save_path: str = "reports/figures/feature_boxplots.png") -> plt.Figure:
    """Plot small-multiple grid of boxplots for every numeric audio feature split by charted."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    for i, feat in enumerate(AUDIO_FEATURES):
        sns.boxplot(x='charted', y=feat, hue='charted', data=df, ax=axes[i], palette=['#7F8C8D', '#26407A'], width=0.5, fliersize=2, legend=False)
        axes[i].set_title(feat.replace('_', ' ').title(), fontsize=11)
        axes[i].set_xlabel("Charted Status", fontsize=9)
        axes[i].set_ylabel(feat, fontsize=9)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Not Charted (0)', 'Charted (1)'])
        axes[i].grid(True, linestyle='--', alpha=0.4)
    display_and_save(fig, save_path)
    return fig


def compute_target_correlations(df: pd.DataFrame, save_path: str = "reports/tables/target_correlations.csv") -> pd.DataFrame:
    """Calculate point-biserial correlation between each numeric audio feature and charted."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rows = []
    for feat in AUDIO_FEATURES:
        r, p = stats.pointbiserialr(df['charted'], df[feat])
        rows.append({'feature': feat, 'correlation': r, 'p_value': p, 'abs_r': abs(r)})
    corr_df = pd.DataFrame(rows).sort_values(by='abs_r', ascending=False).drop(columns=['abs_r'])
    corr_df.to_csv(save_path, index=False)
    return corr_df


def run_eda(data_path: str = "data/processed/songs.parquet", full_path: str = "data/processed/songs_full.parquet") -> dict:
    """Execute complete EDA pipeline and generate all figures and tables."""
    df = pd.read_parquet(data_path)
    df_full = pd.read_parquet(full_path)

    plot_class_balance_natural(df_full)
    plot_class_balance_modeling(df)
    plot_tempo_distribution(df)
    plot_valence_distribution(df)
    plot_danceability_vs_energy(df)
    plot_correlation_heatmap(df)
    plot_duration_by_year(df)
    plot_feature_boxplots(df)

    corr_df = compute_target_correlations(df)
    charted_duration_count = int(df[df['charted'] == 1]['chart_year'].notnull().sum())

    summary = {
        'total_rows': len(df),
        'charted_count': int((df['charted'] == 1).sum()),
        'non_charted_count': int((df['charted'] == 0).sum()),
        'charted_pct': float((df['charted'] == 1).mean() * 100),
        'non_charted_pct': float((df['charted'] == 0).mean() * 100),
        'audio_features': AUDIO_FEATURES,
        'missing_values': df[AUDIO_FEATURES].isnull().sum().to_dict(),
        'chart_year_has_missing': bool(df['chart_year'].isnull().any()),
        'chart_year_missing_count': int(df['chart_year'].isnull().sum()),
        'charted_obs_duration_by_year': charted_duration_count,
        'correlations': corr_df
    }
    return summary


def main() -> None:
    """CLI entry point for EDA pipeline."""
    print("=== Hit Song Predictor: Exploratory Data Analysis ===")
    res = run_eda()

    print(f"\nTotal rows used: {res['total_rows']}")
    print(f"Charted count: {res['charted_count']}")
    print(f"Non-charted count: {res['non_charted_count']}")
    print(f"Charted percentage: {res['charted_pct']:.2f}%")
    print(f"Non-charted percentage: {res['non_charted_pct']:.2f}%")
    print(f"Numeric audio feature list: {res['audio_features']}")
    print(f"Missing values remaining in each modelling feature: {res['missing_values']}")
    print(f"Whether chart_year contains missing values: {res['chart_year_has_missing']} ({res['chart_year_missing_count']} missing)")
    print(f"Number of charted observations used in duration_by_year: {res['charted_obs_duration_by_year']}")

    print("\n--- Point-Biserial Target Correlations ---")
    print(res['correlations'].to_string(index=False))


if __name__ == "__main__":
    main()
