"""
Script de Coleta de Dados - Tech Challenge Fase 4
Pessoa 1: Coleta de dados históricos usando yfinance
"""

import yfinance as yf
import pandas as pd
import sys

from config import (
    STOCK_SYMBOL, START_DATE, END_DATE, 
    RAW_DATA_FILE, VERBOSE
)

def download_stock_data(symbol, start_date, end_date):
    """
    Baixa dados históricos de ações usando yfinance
    
    Args:
        symbol (str): Símbolo da ação (ex: VALE3)
        start_date (str): Data inicial (YYYY-MM-DD)
        end_date (str): Data final (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: DataFrame com dados históricos
    """
    print(f"\n{'='*60}")
    print("📡 COLETANDO DADOS DO YAHOO FINANCE")
    print(f"{'='*60}")
    print(f"   • Empresa: {symbol}")
    print(f"   • Período: {start_date} até {end_date}")
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            print(f"❌ Erro: Nenhum dado foi retornado para {symbol}")
            sys.exit(1)
        
        print(f"✅ Dados coletados: {len(df)} registros")
        return df
    
    except Exception as e:
        print(f"❌ Erro ao baixar dados: {str(e)}")
        sys.exit(1)

def validate_data(df):
    """
        Valida os dados coletados
    
    Args:
        df (pd.DataFrame): DataFrame com dados
    
    Returns:
        bool: True se dados válidos
    """
    print(f"\n{'='*60}")
    print("🔍 VALIDANDO DADOS")
    print(f"{'='*60}")
    
    issues = []

    # Corrige possíveis colunas MultiIndex
    df.columns = df.columns.get_level_values(0)
    
    # Verificar colunas essenciais
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Colunas faltando: {missing_cols}")
    
    # Verificar valores nulos
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        issues.append(f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}")
    
    # Verificar valores negativos
    for col in required_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if isinstance(neg_count, pd.Series):
                neg_count = neg_count.sum()
            if neg_count > 0:
                issues.append(f"Valores negativos em {col}: {neg_count}")
    
    # Verificar duplicatas
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        issues.append(f"Datas duplicadas: {dup_count}")
    
    if issues:
        print("⚠️  Problemas encontrados:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ Dados válidos! Nenhum problema encontrado.")
        return True


def save_raw_data(df, filepath):
    """
    Salva dados brutos em CSV
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        filepath (Path): Caminho do arquivo
    """
    print(f"\n{'='*60}")
    print("💾 SALVANDO DADOS BRUTOS")
    print(f"{'='*60}")
    
    try:
        df.to_csv(filepath)
        print(f"✅ Dados salvos em: {filepath}")
        print(f"   • Tamanho: {filepath.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ Erro ao salvar dados: {str(e)}")
        sys.exit(1)

def print_data_summary(df):
    """
    Imprime resumo dos dados coletados
    
    Args:
        df (pd.DataFrame): DataFrame com dados
    """
    print(f"\n{'='*60}")
    print("📊 RESUMO DOS DADOS")
    print(f"{'='*60}")
    print(f"\n🔢 Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"\n📅 Período:")
    print(f"   • Início: {df.index.min().date()}")
    print(f"   • Fim: {df.index.max().date()}")
    print(f"   • Total de dias: {(df.index.max() - df.index.min()).days}")
    
    print(f"\n💰 Estatísticas do Close (R$):")
    print(f"   • Mínimo: R$ {df['Close'].min():.2f}")
    print(f"   • Máximo: R$ {df['Close'].max():.2f}")
    print(f"   • Média: R$ {df['Close'].mean():.2f}")
    print(f"   • Último: R$ {df['Close'].iloc[-1]:.2f}")
    
    print(f"\n📦 Volume:")
    print(f"   • Média diária: {df['Volume'].mean():,.0f}")
    print(f"   • Máximo: {df['Volume'].max():,.0f}")
    
    print(f"\n📋 Colunas disponíveis:")
    for col in df.columns:
        print(f"   • {col}")

def main():
    """
    Função principal de coleta de dados
    """
    print(f"\n{'#'*60}")
    print("# TECH CHALLENGE FASE 4 - COLETA DE DADOS")
    print(f"{'#'*60}")
    
    # 1. Baixar dados
    df = download_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
    
    # 2. Validar dados
    is_valid = validate_data(df)
    if not is_valid:
        print("\n⚠️  Aviso: Dados contêm problemas, mas serão salvos para tratamento posterior.")
    
    # 3. Salvar dados brutos
    save_raw_data(df, RAW_DATA_FILE)
    
    # 4. Mostrar resumo
    print_data_summary(df)
    
    print(f"\n{'='*60}")
    print("✅ COLETA DE DADOS CONCLUÍDA COM SUCESSO!")
    print(f"{'='*60}")
    print(f"\n📂 Próximo passo: Execute 'python src/data_preprocessing.py'")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()