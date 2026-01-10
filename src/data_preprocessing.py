"""
Script de Pré-processamento - Tech Challenge Fase 4
Pessoa 1: Limpeza e engenharia de features
"""

import pandas as pd
import numpy as np
import sys

from config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE,
    ADD_TECHNICAL_INDICATORS, SMA_PERIODS, EMA_PERIODS,
    RSI_PERIOD, MACD_PARAMS, BOLLINGER_PERIOD, BOLLINGER_STD,
    OUTLIER_METHOD, OUTLIER_THRESHOLD
)

def load_raw_data(filepath):
    """
    Carrega dados brutos
    
    Args:
        filepath (Path): Caminho do arquivo CSV
    
    Returns:
        pd.DataFrame: DataFrame com dados brutos
    """
    print(f"\n{'='*60}")
    print("📂 CARREGANDO DADOS BRUTOS")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✅ Dados carregados: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {str(e)}")
        sys.exit(1)

def handle_missing_values(df):
    """
    Trata valores ausentes
    
    Args:
        df (pd.DataFrame): DataFrame com dados
    
    Returns:
        pd.DataFrame: DataFrame sem valores nulos
    """
    print(f"\n{'='*60}")
    print("🔧 TRATANDO VALORES AUSENTES")
    print(f"{'='*60}")
    
    null_before = df.isnull().sum().sum()
    print(f"   • Valores nulos antes: {null_before}")
    
    if null_before > 0:
        # Interpolação linear para valores numéricos
        df = df.interpolate(method='linear', limit_direction='both')
        
        # Remove linhas que ainda têm nulos (início/fim)
        df = df.dropna()
        
        null_after = df.isnull().sum().sum()
        print(f"   • Valores nulos depois: {null_after}")
        print(f"   • Registros removidos: {null_before - null_after}")
    else:
        print("   • Nenhum valor nulo encontrado")
    
    return df

def detect_and_handle_outliers(df, columns=['Open', 'High', 'Low', 'Close'], method='IQR', threshold=1.5):
    """
    Detecta e trata outliers usando método IQR
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        columns (list): Colunas para detectar outliers
        method (str): Método de detecção
        threshold (float): Multiplicador do IQR
    
    Returns:
        pd.DataFrame: DataFrame sem outliers extremos
    """
    print(f"\n{'='*60}")
    print(f"🎯 DETECTANDO OUTLIERS ({method})")
    print(f"{'='*60}")
    
    df_clean = df.copy()
    outliers_count = 0
    
    for col in columns:
        if col not in df.columns:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Contar outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers_count += outliers
        
        if outliers > 0:
            print(f"   • {col}: {outliers} outliers detectados")
            # Substituir outliers pelos limites (winsorização)
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"\n✅ Total de outliers tratados: {outliers_count}")
    return df_clean

def add_technical_indicators(df):
    """
    Adiciona indicadores técnicos
    
    Args:
        df (pd.DataFrame): DataFrame com dados
    
    Returns:
        pd.DataFrame: DataFrame com indicadores técnicos
    """
    print(f"\n{'='*60}")
    print("📈 ADICIONANDO INDICADORES TÉCNICOS")
    print(f"{'='*60}")
    
    df_tech = df.copy()
    
    # 1. Médias Móveis Simples (SMA)
    for period in SMA_PERIODS:
        df_tech[f'SMA_{period}'] = df_tech['Close'].rolling(window=period).mean()
        print(f"   ✅ SMA_{period}")
    
    # 2. Médias Móveis Exponenciais (EMA)
    for period in EMA_PERIODS:
        df_tech[f'EMA_{period}'] = df_tech['Close'].ewm(span=period, adjust=False).mean()
        print(f"   ✅ EMA_{period}")
    
    # 3. RSI (Relative Strength Index)
    delta = df_tech['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df_tech['RSI'] = 100 - (100 / (1 + rs))
    print(f"   ✅ RSI")
    
    # 4. MACD
    ema_fast = df_tech['Close'].ewm(span=MACD_PARAMS['fast'], adjust=False).mean()
    ema_slow = df_tech['Close'].ewm(span=MACD_PARAMS['slow'], adjust=False).mean()
    df_tech['MACD'] = ema_fast - ema_slow
    df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=MACD_PARAMS['signal'], adjust=False).mean()
    df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['MACD_Signal']
    print(f"   ✅ MACD, MACD_Signal, MACD_Hist")
    
    # 5. Bollinger Bands
    sma = df_tech['Close'].rolling(window=BOLLINGER_PERIOD).mean()
    std = df_tech['Close'].rolling(window=BOLLINGER_PERIOD).std()
    df_tech['BB_Upper'] = sma + (BOLLINGER_STD * std)
    df_tech['BB_Lower'] = sma - (BOLLINGER_STD * std)
    df_tech['BB_Middle'] = sma
    print(f"   ✅ BB_Upper, BB_Middle, BB_Lower")
    
    # 6. Volatilidade (Desvio padrão de 20 dias)
    df_tech['Volatility'] = df_tech['Close'].rolling(window=20).std()
    print(f"   ✅ Volatility")
    
    # 7. Retorno diário
    df_tech['Daily_Return'] = df_tech['Close'].pct_change()
    print(f"   ✅ Daily_Return")
    
    # Remove linhas com NaN gerados pelos indicadores
    df_tech = df_tech.dropna()
    
    print(f"\n✅ Total de features: {len(df_tech.columns)}")
    
    return df_tech

def validate_processed_data(df):
    """
    Valida dados processados
    
    Args:
        df (pd.DataFrame): DataFrame processado
    
    Returns:
        bool: True se válido
    """
    print(f"\n{'='*60}")
    print("✔️  VALIDANDO DADOS PROCESSADOS")
    print(f"{'='*60}")
    
    issues = []
    
    # Verificar valores nulos
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        issues.append(f"Valores nulos: {null_count}")
    
    # Verificar valores infinitos
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        issues.append(f"Valores infinitos: {inf_count}")
    
    # Verificar se há dados suficientes
    if len(df) < 100:
        issues.append(f"Poucos registros: {len(df)} (mínimo 100)")
    
    if issues:
        print("⚠️  Problemas encontrados:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ Dados processados válidos!")
        print(f"   • Registros: {len(df)}")
        print(f"   • Features: {len(df.columns)}")
        print(f"   • Sem valores nulos ou infinitos")
        return True

def save_processed_data(df, filepath):
    """
    Salva dados processados
    
    Args:
        df (pd.DataFrame): DataFrame processado
        filepath (Path): Caminho do arquivo
    """
    print(f"\n{'='*60}")
    print("💾 SALVANDO DADOS PROCESSADOS")
    print(f"{'='*60}")
    
    try:
        df.to_csv(filepath)
        print(f"✅ Dados salvos em: {filepath}")
        print(f"   • Tamanho: {filepath.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"❌ Erro ao salvar dados: {str(e)}")
        sys.exit(1)

def print_processing_summary(df_before, df_after):
    """
    Imprime resumo do processamento
    
    Args:
        df_before (pd.DataFrame): DataFrame antes
        df_after (pd.DataFrame): DataFrame depois
    """
    print(f"\n{'='*60}")
    print("📊 RESUMO DO PRÉ-PROCESSAMENTO")
    print(f"{'='*60}")
    print(f"\n📉 Antes:")
    print(f"   • Registros: {len(df_before)}")
    print(f"   • Features: {len(df_before.columns)}")
    
    print(f"\n📈 Depois:")
    print(f"   • Registros: {len(df_after)}")
    print(f"   • Features: {len(df_after.columns)}")
    
    print(f"\n🔄 Mudanças:")
    print(f"   • Registros removidos: {len(df_before) - len(df_after)}")
    print(f"   • Features adicionadas: {len(df_after.columns) - len(df_before.columns)}")
    
    print(f"\n📋 Features finais:")
    for i, col in enumerate(df_after.columns, 1):
        print(f"   {i:2d}. {col}")

def main():
    """
    Função principal de pré-processamento
    """
    print(f"\n{'#'*60}")
    print("# TECH CHALLENGE FASE 4 - PRÉ-PROCESSAMENTO")
    print(f"{'#'*60}")
    
    # 1. Carregar dados brutos
    df = load_raw_data(RAW_DATA_FILE)
    df_original = df.copy()
    
    # 2. Tratar valores ausentes
    df = handle_missing_values(df)
    
    # 3. Detectar e tratar outliers
    df = detect_and_handle_outliers(df, method=OUTLIER_METHOD, threshold=OUTLIER_THRESHOLD)
    
    # 4. Adicionar indicadores técnicos
    if ADD_TECHNICAL_INDICATORS:
        df = add_technical_indicators(df)
    
    # 5. Validar dados processados
    is_valid = validate_processed_data(df)
    if not is_valid:
        print("\n❌ Dados processados contêm problemas. Verifique os erros acima.")
        sys.exit(1)
    
    # 6. Salvar dados processados
    save_processed_data(df, PROCESSED_DATA_FILE)
    
    # 7. Mostrar resumo
    print_processing_summary(df_original, df)
    
    print(f"\n{'='*60}")
    print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}")
    print(f"\n📂 Próximo passo: Execute 'python src/data_preparation.py'")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()