"""
Script de Preparação de Dados para LSTM - Tech Challenge Fase 4
Pessoa 1: Normalização, criação de janelas temporais e divisão treino/val/teste
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
from sklearn.preprocessing import MinMaxScaler

from config import (
    PROCESSED_DATA_FILE, LOOKBACK_DAYS, FORECAST_HORIZON,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, TARGET,
    SCALER_FILE, TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE,
    DATA_INFO_FILE, FEATURE_RANGE, RANDOM_STATE
)

def load_processed_data(filepath):
    """
    Carrega os dados processados
    
    Args:
        filepath (str): Caminho do arquivo CSV
    
    Returns:
        pd.DataFrame: DataFrame processado
    """
    print(f"\n{'='*60}")
    print("📂 CARREGANDO DADOS PROCESSADOS")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✅ Dados carregados: {df.shape}")
        print(f"   • Features: {len(df.columns)}")
        print(f"   • Registros: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {str(e)}")
        sys.exit(1)

def normalize_data(df):
    """
    Normaliza os dados usando MinMaxScaler
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
    
    Returns:
        tuple: (dados normalizados, scaler, feature_names)
    """
    print(f"\n{'='*60}")
    print("🔄 NORMALIZANDO DADOS")
    print(f"{'='*60}")
    
    # Criar scaler
    scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
    
    # Guardar nomes das features
    feature_names = df.columns.tolist()
    
    # Normalizar os dados
    data_normalized = scaler.fit_transform(df.values)
    
    print(f"✅ Dados normalizados para o intervalo {FEATURE_RANGE}")
    print(f"   • Shape: {data_normalized.shape}")
    print(f"   • Min: {data_normalized.min():.4f}")
    print(f"   • Max: {data_normalized.max():.4f}")
    
    # Salvar scaler para uso futuro
    try:
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler salvo em: {SCALER_FILE}")
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível salvar o scaler: {str(e)}")
    
    return data_normalized, scaler, feature_names

def create_sequences(data, feature_names, lookback=60, forecast_horizon=1, target_col='Close'):
    """
    Cria sequências temporais para treinar o LSTM
    
    Args:
        data (np.array): Dados normalizados
        feature_names (list): Lista com nomes das features
        lookback (int): Número de dias anteriores (janela temporal)
        forecast_horizon (int): Dias à frente para prever
        target_col (str): Nome da coluna target
    
    Returns:
        tuple: (X, y) - Arrays com sequências de entrada e saída
    """
    print(f"\n{'='*60}")
    print("🔨 CRIANDO SEQUÊNCIAS TEMPORAIS")
    print(f"{'='*60}")
    print(f"   • Lookback: {lookback} dias")
    print(f"   • Forecast horizon: {forecast_horizon} dia(s)")
    print(f"   • Target: {target_col}")
    
    X, y = [], []
    
    # Encontrar índice da coluna target
    target_idx = feature_names.index(target_col)
    
    # Criar sequências
    for i in range(lookback, len(data) - forecast_horizon + 1):
        # X: últimos 'lookback' dias com todas as features
        X.append(data[i - lookback:i, :])
        
        # y: preço de fechamento do próximo dia (ou dias, se forecast_horizon > 1)
        y.append(data[i + forecast_horizon - 1, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Sequências criadas:")
    print(f"   • X shape: {X.shape} (samples, lookback, features)")
    print(f"   • y shape: {y.shape} (samples,)")
    print(f"   • Total de sequências: {len(X)}")
    
    return X, y

def split_data(X, y, train_size=0.70, val_size=0.15, test_size=0.15):
    """
    Divide os dados em treino, validação e teste (séries temporais)
    
    Args:
        X (np.array): Features (sequências)
        y (np.array): Target
        train_size (float): Proporção de treino
        val_size (float): Proporção de validação
        test_size (float): Proporção de teste
    
    Returns:
        dict: Dicionário com os dados divididos
    """
    print(f"\n{'='*60}")
    print("✂️  DIVIDINDO DADOS EM TREINO/VALIDAÇÃO/TESTE")
    print(f"{'='*60}")
    
    # Verificar se as proporções somam 1
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        print(f"⚠️  Aviso: Proporções somam {total:.2f}, ajustando...")
        train_size = train_size / total
        val_size = val_size / total
        test_size = test_size / total
    
    n_samples = len(X)
    
    # Calcular índices de corte (respeitando ordem temporal)
    train_end = int(n_samples * train_size)
    val_end = int(n_samples * (train_size + val_size))
    
    # Dividir os dados
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"📊 Divisão dos dados:")
    print(f"   • Treino:     {len(X_train):5d} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"   • Validação:  {len(X_val):5d} samples ({len(X_val)/n_samples*100:.1f}%)")
    print(f"   • Teste:      {len(X_test):5d} samples ({len(X_test)/n_samples*100:.1f}%)")
    print(f"   • Total:      {n_samples:5d} samples")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

def save_split_data(data_dict, train_file, val_file, test_file):
    """
    Salva os dados divididos em arquivos numpy
    
    Args:
        data_dict (dict): Dicionário com os dados divididos
        train_file (str): Arquivo para dados de treino
        val_file (str): Arquivo para dados de validação
        test_file (str): Arquivo para dados de teste
    """
    print(f"\n{'='*60}")
    print("💾 SALVANDO DADOS DIVIDIDOS")
    print(f"{'='*60}")
    
    try:
        # Salvar treino
        np.savez_compressed(
            train_file,
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train']
        )
        print(f"✅ Treino salvo em: {train_file}")
        
        # Salvar validação
        np.savez_compressed(
            val_file,
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val']
        )
        print(f"✅ Validação salvo em: {val_file}")
        
        # Salvar teste
        np.savez_compressed(
            test_file,
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test']
        )
        print(f"✅ Teste salvo em: {test_file}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar dados: {str(e)}")
        sys.exit(1)

def save_data_info(df_original, feature_names, data_dict, scaler):
    """
    Salva informações sobre os dados processados em JSON
    
    Args:
        df_original (pd.DataFrame): DataFrame original
        feature_names (list): Lista de features
        data_dict (dict): Dados divididos
        scaler: Scaler utilizado
    """
    print(f"\n{'='*60}")
    print("📝 SALVANDO INFORMAÇÕES DOS DADOS")
    print(f"{'='*60}")
    
    info = {
        'dataset_info': {
            'total_records': len(df_original),
            'date_range': {
                'start': str(df_original.index.min().date()),
                'end': str(df_original.index.max().date())
            },
            'features': feature_names,
            'n_features': len(feature_names),
            'target': TARGET
        },
        'preprocessing': {
            'lookback_days': LOOKBACK_DAYS,
            'forecast_horizon': FORECAST_HORIZON,
            'normalization': {
                'method': 'MinMaxScaler',
                'feature_range': FEATURE_RANGE
            }
        },
        'data_split': {
            'train': {
                'samples': int(len(data_dict['X_train'])),
                'percentage': float(TRAIN_SIZE * 100)
            },
            'validation': {
                'samples': int(len(data_dict['X_val'])),
                'percentage': float(VAL_SIZE * 100)
            },
            'test': {
                'samples': int(len(data_dict['X_test'])),
                'percentage': float(TEST_SIZE * 100)
            }
        },
        'shapes': {
            'X_train': list(data_dict['X_train'].shape),
            'y_train': list(data_dict['y_train'].shape),
            'X_val': list(data_dict['X_val'].shape),
            'y_val': list(data_dict['y_val'].shape),
            'X_test': list(data_dict['X_test'].shape),
            'y_test': list(data_dict['y_test'].shape)
        }
    }
    
    try:
        with open(DATA_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"✅ Informações salvas em: {DATA_INFO_FILE}")
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível salvar informações: {str(e)}")

def main():
    """
    Função principal de preparação dos dados
    """
    print(f"\n{'#'*60}")
    print(f"# TECH CHALLENGE FASE 4 - PREPARAÇÃO DOS DADOS")
    print(f"{'#'*60}")
    
    # 1. Carregar dados processados
    df = load_processed_data(PROCESSED_DATA_FILE)
    
    # 2. Normalizar dados
    data_normalized, scaler, feature_names = normalize_data(df)
    
    # 3. Criar sequências temporais
    X, y = create_sequences(
        data_normalized, 
        feature_names, 
        lookback=LOOKBACK_DAYS,
        forecast_horizon=FORECAST_HORIZON,
        target_col=TARGET
    )
    
    # 4. Dividir dados em treino/validação/teste
    data_dict = split_data(X, y, TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
    
    # 5. Salvar dados divididos
    save_split_data(data_dict, TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE)
    
    # 6. Salvar informações dos dados
    save_data_info(df, feature_names, data_dict, scaler)
    
    print(f"\n{'='*60}")
    print("✅ PREPARAÇÃO DOS DADOS CONCLUÍDA COM SUCESSO!")
    print(f"{'='*60}")
    print(f"\n🎉 DADOS PRONTOS PARA A PESSOA 2 (TREINAMENTO DO MODELO)")
    print(f"\n📦 Arquivos gerados:")
    print(f"   • {TRAIN_DATA_FILE}")
    print(f"   • {VAL_DATA_FILE}")
    print(f"   • {TEST_DATA_FILE}")
    print(f"   • {SCALER_FILE}")
    print(f"   • {DATA_INFO_FILE}")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()