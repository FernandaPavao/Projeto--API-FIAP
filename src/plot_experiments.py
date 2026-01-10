# plot_experiments.py
"""
Gera gráficos e ranking a partir de results/experiment_log.csv

Saída:
 - results/experiments_mae.png
 - results/experiments_rmse.png
 - results/experiments_mape.png
 - results/experiment_ranking.csv
 - results/experiments_summary.txt
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Config
BASE_DIR = Path('.')
RESULTS_DIR = BASE_DIR / 'results'
CSV_PATH = RESULTS_DIR / 'experiment_log.csv'

# Saídas
OUT_MAE = RESULTS_DIR / 'experiments_mae.png'
OUT_RMSE = RESULTS_DIR / 'experiments_rmse.png'
OUT_MAPE = RESULTS_DIR / 'experiments_mape.png'
OUT_RANK = RESULTS_DIR / 'experiment_ranking.csv'
OUT_SUMMARY = RESULTS_DIR / 'experiments_summary.txt'

def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_experiments(path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    # Normalizar nomes de colunas (lower)
    df.columns = [c.strip() for c in df.columns]
    expected = {'model','units1','units2','dropout','batch','lr','epochs','mae','rmse','mape','notes'}
    # try to convert some columns to numeric if exist
    for col in ['units1','units2','dropout','batch','lr','epochs','mae','rmse','mape']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_metric(df, metric, outpath, title=None):
    # Create readable labels for experiments
    labels = []
    for idx, row in df.iterrows():
        label = f"{row.get('model','exp')}_u{int(row['units1']) if not np.isnan(row.get('units1', np.nan)) else 'NA'}"
        if 'units2' in row and not pd.isna(row['units2']):
            label += f"-{int(row['units2'])}"
        labels.append(label)
    values = df[metric].values
    x = np.arange(len(values))

    plt.figure(figsize=(max(8, len(values)*0.6), 6))
    bars = plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(metric.upper())
    plt.title(title or f'Experiments - {metric.upper()}')
    plt.grid(axis='y', alpha=0.25)

    # Annotate bars
    for rect, val in zip(bars, values):
        if not (pd.isna(val)):
            plt.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), f"{val:.3f}",
                     ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"✅ Saved {outpath}")

def generate_ranking(df, by='mae'):
    if by not in df.columns:
        raise ValueError(f"Coluna {by} não existe no CSV")
    rank_df = df.sort_values(by=by, ascending=True).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    rank_df.index.name = 'rank'
    return rank_df

def save_summary(rank_df, out_txt, top_k=3):
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("Experiments ranking summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total experiments: {len(rank_df)}\n\n")
        f.write("Top experiments (by MAE):\n")
        f.write("-"*60 + "\n")
        for i, row in rank_df.head(top_k).iterrows():
            f.write(f"Rank {i}: model={row.get('model')} | units1={row.get('units1')} | units2={row.get('units2')} | dropout={row.get('dropout')} | batch={row.get('batch')} | lr={row.get('lr')} | mae={row.get('mae'):.6f} | rmse={row.get('rmse'):.6f} | mape={row.get('mape'):.4f}%\n")
        f.write("\nFull ranking saved to experiment_ranking.csv\n")
    print(f"✅ Saved summary to {out_txt}")

def main():
    ensure_results_dir()
    df = load_experiments(CSV_PATH)
    if df.empty:
        print("Nenhum experimento encontrado em experiment_log.csv")
        return

    # Fill missing notes column if not present
    if 'notes' not in df.columns:
        df['notes'] = ''

    # Ensure numeric columns exist and fill NaNs with large value to push them to bottom
    for col in ['mae','rmse','mape']:
        if col not in df.columns:
            df[col] = np.nan

    # Plot metrics
    try:
        plot_metric(df, 'mae', OUT_MAE, title='MAE por Experimento')
    except Exception as e:
        print("Erro plot MAE:", e)
    try:
        plot_metric(df, 'rmse', OUT_RMSE, title='RMSE por Experimento')
    except Exception as e:
        print("Erro plot RMSE:", e)
    try:
        plot_metric(df, 'mape', OUT_MAPE, title='MAPE por Experimento')
    except Exception as e:
        print("Erro plot MAPE:", e)

    # Ranking (by MAE primary)
    rank_df = generate_ranking(df, by='mae')
    rank_df.to_csv(OUT_RANK, index=True)
    print(f"✅ Saved ranking CSV: {OUT_RANK}")

    # Save textual summary
    save_summary(rank_df, OUT_SUMMARY, top_k=3)

    print("\n🎯 Done. Files created in results/:")
    print(f" - {OUT_MAE.name}")
    print(f" - {OUT_RMSE.name}")
    print(f" - {OUT_MAPE.name}")
    print(f" - {OUT_RANK.name}")
    print(f" - {OUT_SUMMARY.name}")

if __name__ == '__main__':
    main()
