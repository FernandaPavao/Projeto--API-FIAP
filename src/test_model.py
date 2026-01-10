# test_model.py
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# CONFIG
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
REPORTS_DIR = BASE_DIR / 'reports' / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / 'best_model.keras'

# UTIL: carregar dados npz
def load_npz(file_path):
    arr = np.load(file_path)
    return arr

# UTIL: desnormalizar apenas a coluna target
def denormalize_target(scaler, y_norm, feature_names, target_name='Close'):
    idx = feature_names.index(target_name)
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(y_norm), n_features))
    dummy[:, idx] = y_norm.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, idx]

def main():
    # Carregar modelo
    print("Carregando modelo...")
    model = load_model(MODEL_PATH)

    # Carregar scaler e info
    with open(DATA_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(DATA_DIR / 'data_info.json', 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    feature_names = data_info['dataset_info']['features']
    target = data_info['dataset_info']['target']

    # Carregar dados de teste
    test_data = load_npz(DATA_DIR / 'test_data.npz')
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Fazer previsões
    print("Fazendo previsões...")
    y_pred_norm = model.predict(X_test).ravel()
    y_test_norm = y_test.ravel()

    # Desnormalizar
    y_pred = denormalize_target(scaler, y_pred_norm, feature_names, target_name=target)
    y_true = denormalize_target(scaler, y_test_norm, feature_names, target_name=target)

    # Calcular métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Salvar métricas
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    with open(RESULTS_DIR / 'metrics_test.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Salvar previsões
    results_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    results_df.to_csv(RESULTS_DIR / 'predictions.csv', index=False)

    # Gráfico real vs previsto
    plt.figure(figsize=(12,6))
    plt.plot(y_true, label='Real', color='blue')
    plt.plot(y_pred, label='Previsto', color='red', alpha=0.7)
    plt.title('Previsão vs Real')
    plt.xlabel('Amostras')
    plt.ylabel('Preço R$')
    plt.legend()
    plt.savefig(REPORTS_DIR / 'pred_vs_real.png')
    plt.show()

    # Histograma de erros
    plt.figure(figsize=(10,5))
    errors = y_true - y_pred
    plt.hist(errors, bins=50, color='purple', alpha=0.7)
    plt.title('Distribuição dos Erros')
    plt.xlabel('Erro')
    plt.ylabel('Frequência')
    plt.savefig(REPORTS_DIR / 'error_distribution.png')
    plt.show()

if __name__ == '__main__':
    main()
