"""
Script de Análise Exploratória - Tech Challenge Fase 4
Pessoa 1: EDA (Exploratory Data Analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from config import (
    PROCESSED_DATA_FILE, FIGURES_DIR,
    FIGURE_SIZE, DPI
)

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_processed_data(filepath):
    """
    Carrega dados processados
    
    Args:
        filepath (Path): Caminho do arquivo
    
    Returns:
        pd.DataFrame: DataFrame processado
    """
    print(f"\n{'='*60}")
    print("📂 CARREGANDO DADOS PROCESSADOS")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✅ Dados carregados: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {str(e)}")
        sys.exit(1)

def plot_price_history(df):
    """
    Gráfico de histórico de preços e volume
    """
    print("\n📊 Gerando gráfico: Histórico de Preços e Volume...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Gráfico de preços
    ax1.plot(df.index, df['Close'], label='Close', linewidth=1.5, color='#2E86AB')
    ax1.fill_between(df.index, df['Low'], df['High'], alpha=0.2, color='#A23B72')
    ax1.set_ylabel('Preço (R$)', fontsize=12)
    ax1.set_title('Histórico de Preços - VALE3.SA', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de volume
    ax2.bar(df.index, df['Volume'], alpha=0.6, color='#F18F01')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.set_title('Volume de Negociação', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / '01_price_history.png'
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: {filepath}")

def plot_returns_distribution(df):
    """
    Gráfico de distribuição de retornos
    """
    print("\n📊 Gerando gráfico: Distribuição de Retornos...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    ax1.hist(df['Daily_Return'].dropna(), bins=50, alpha=0.7, color='#06A77D', edgecolor='black')
    ax1.axvline(df['Daily_Return'].mean(), color='red', linestyle='--', label=f'Média: {df["Daily_Return"].mean():.4f}')
    ax1.set_xlabel('Retorno Diário', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.set_title('Distribuição de Retornos Diários', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(df['Daily_Return'].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor='#06A77D', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Retorno Diário', fontsize=12)
    ax2.set_title('Box Plot - Retornos Diários', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / '02_returns_distribution.png'
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: {filepath}")

def plot_moving_averages(df):
    """
    Gráfico de médias móveis
    """
    print("\n📊 Gerando gráfico: Médias Móveis...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df.index, df['Close'], label='Close', linewidth=1.5, alpha=0.8, color='#2E86AB')
    
    # Plotar SMAs se existirem
    colors = ['#F18F01', '#06A77D', '#A23B72']
    for i, col in enumerate([c for c in df.columns if c.startswith('SMA_')]):
        ax.plot(df.index, df[col], label=col, linewidth=1.2, alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Preço (R$)', fontsize=12)
    ax.set_title('Preço de Fechamento com Médias Móveis', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / '03_moving_averages.png'
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: {filepath}")

def plot_technical_indicators(df):
    """
    Gráfico de indicadores técnicos
    """
    print("\n📊 Gerando gráfico: Indicadores Técnicos...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # RSI
    if 'RSI' in df.columns:
        axes[0].plot(df.index, df['RSI'], label='RSI', linewidth=1.5, color='#2E86AB')
        axes[0].axhline(70, color='red', linestyle='--', alpha=0.5, label='Sobrecomprado')
        axes[0].axhline(30, color='green', linestyle='--', alpha=0.5, label='Sobrevendido')
        axes[0].set_ylabel('RSI', fontsize=12)
        axes[0].set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
    
    # MACD
    if 'MACD' in df.columns:
        axes[1].plot(df.index, df['MACD'], label='MACD', linewidth=1.5, color='#F18F01')
        axes[1].plot(df.index, df['MACD_Signal'], label='Signal', linewidth=1.5, color='#06A77D')
        axes[1].bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.3, color='#A23B72')
        axes[1].set_ylabel('MACD', fontsize=12)
        axes[1].set_title('Moving Average Convergence Divergence (MACD)', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        axes[2].plot(df.index, df['Close'], label='Close', linewidth=1.5, color='#2E86AB')
        axes[2].plot(df.index, df['BB_Upper'], label='BB Upper', linewidth=1, linestyle='--', color='red', alpha=0.7)
        axes[2].plot(df.index, df['BB_Middle'], label='BB Middle', linewidth=1, linestyle='--', color='gray', alpha=0.7)
        axes[2].plot(df.index, df['BB_Lower'], label='BB Lower', linewidth=1, linestyle='--', color='green', alpha=0.7)
        axes[2].fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.1, color='gray')
        axes[2].set_ylabel('Preço (R$)', fontsize=12)
        axes[2].set_xlabel('Data', fontsize=12)
        axes[2].set_title('Bollinger Bands', fontsize=14, fontweight='bold')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / '04_technical_indicators.png'
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: {filepath}")

def plot_correlation_matrix(df):
    """
    Matriz de correlação
    """
    print("\n📊 Gerando gráfico: Matriz de Correlação...")
    
    # Selecionar apenas colunas numéricas principais
    cols_to_plot = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Adicionar indicadores se existirem
    for col in df.columns:
        if any(indicator in col for indicator in ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'Volatility']):
            cols_to_plot.append(col)
    
    # Limitar a 15 features para visualização
    cols_to_plot = cols_to_plot[:15]
    
    corr_matrix = df[cols_to_plot].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title('Matriz de Correlação - Features Principais', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / '05_correlation_matrix.png'
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Salvo: {filepath}")

def generate_summary_statistics(df):
    """
    Gera estatísticas descritivas
    """
    print("\n📊 Gerando estatísticas descritivas...")
    
    filepath = FIGURES_DIR / 'summary_statistics.txt'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ESTATÍSTICAS DESCRITIVAS - VALE3.SA\n")
        f.write("="*60 + "\n\n")
        
        f.write("INFORMAÇÕES GERAIS\n")
        f.write("-"*60 + "\n")
        f.write(f"Período: {df.index.min().date()} até {df.index.max().date()}\n")
        f.write(f"Total de dias: {len(df)}\n")
        f.write(f"Total de features: {len(df.columns)}\n\n")
        
        f.write("ESTATÍSTICAS DO PREÇO DE FECHAMENTO (R$)\n")
        f.write("-"*60 + "\n")
        f.write(df['Close'].describe().to_string())
        f.write("\n\n")
        
        f.write("ESTATÍSTICAS DO VOLUME\n")
        f.write("-"*60 + "\n")
        f.write(df['Volume'].describe().to_string())
        f.write("\n\n")
        
        if 'Daily_Return' in df.columns:
            f.write("ESTATÍSTICAS DOS RETORNOS DIÁRIOS\n")
            f.write("-"*60 + "\n")
            f.write(df['Daily_Return'].describe().to_string())
            f.write("\n\n")
        
        if 'Volatility' in df.columns:
            f.write("ESTATÍSTICAS DA VOLATILIDADE\n")
            f.write("-"*60 + "\n")
            f.write(df['Volatility'].describe().to_string())
            f.write("\n\n")
        
        f.write("LISTA DE TODAS AS FEATURES\n")
        f.write("-"*60 + "\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:2d}. {col}\n")
    
    print(f"   ✅ Salvo: {filepath}")

def main():
    """
    Função principal de análise exploratória
    """
    print(f"\n{'#'*60}")
    print("# TECH CHALLENGE FASE 4 - ANÁLISE EXPLORATÓRIA (EDA)")
    print(f"{'#'*60}")
    
    # Carregar dados
    df = load_processed_data(PROCESSED_DATA_FILE)
    
    print(f"\n{'='*60}")
    print("📈 GERANDO VISUALIZAÇÕES")
    print(f"{'='*60}")
    
    # Gerar gráficos
    plot_price_history(df)
    plot_returns_distribution(df)
    plot_moving_averages(df)
    plot_technical_indicators(df)
    plot_correlation_matrix(df)
    
    # Gerar estatísticas
    generate_summary_statistics(df)
    
    print(f"\n{'='*60}")
    print("✅ ANÁLISE EXPLORATÓRIA CONCLUÍDA!")
    print(f"{'='*60}")
    print(f"\n📂 Gráficos salvos em: {FIGURES_DIR}")
    print(f"   • 01_price_history.png")
    print(f"   • 02_returns_distribution.png")
    print(f"   • 03_moving_averages.png")
    print(f"   • 04_technical_indicators.png")
    print(f"   • 05_correlation_matrix.png")
    print(f"   • summary_statistics.txt")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()