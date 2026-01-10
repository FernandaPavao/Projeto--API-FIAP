# train_model.py
import os
import json
from pathlib import Path
import numpy as np
import pickle
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# CONFIG
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HYPERPARAMS 
LOOKBACK = None  
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT = 0.3
NOTES= 'More Dropout'

# UTIL: carregar dados
def load_npz(file_path):
    arr = np.load(file_path)
    return arr

def load_data():
    train = load_npz(DATA_DIR / 'train_data.npz')
    val = load_npz(DATA_DIR / 'val_data.npz')
    test = load_npz(DATA_DIR / 'test_data.npz')
    X_train, y_train = train['X_train'], train['y_train']
    X_val, y_val = val['X_val'], val['y_val']
    X_test, y_test = test['X_test'], test['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test

# UTIL: desnormalizar apenas a coluna 'Close' usando scaler e feature index
def denormalize_target(scaler, y_norm, feature_names, target_name='Close'):
    idx = feature_names.index(target_name)
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(y_norm), n_features))
    dummy[:, idx] = y_norm.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, idx]

def main():
    # Carregar info
    info = json.load(open(DATA_DIR / 'data_info.json', 'r', encoding='utf-8'))
    feature_names = info['dataset_info']['features']
    target = info['dataset_info']['target']

    # Carregar dados
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    global LOOKBACK
    LOOKBACK = X_train.shape[1]
    n_features = X_train.shape[2]

    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    # Carregar scaler
    scaler = pickle.load(open(DATA_DIR / 'scaler.pkl', 'rb'))

    # Construir modelo
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(DROPOUT),
        BatchNormalization(),
        LSTM(LSTM_UNITS_2),
        Dropout(DROPOUT),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse', metrics=['mae'])
    model.summary()

    # Callbacks
    ckpt_path = MODELS_DIR / 'best_model.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True)
    ]

    # Treinamento
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    # Salvar modelo
    model.save(ckpt_path)

    print("\n--- PREDIÇÃO & CHECKS ---")

    # garantir dtypes corretos
    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_val   = np.array(y_val).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)

    # 1) Verificar shapes básicos
    print("Shapes (X_train, X_val, X_test):", X_train.shape, X_val.shape, X_test.shape)
    print("Shapes (y_train, y_val, y_test):", y_train.shape, y_val.shape, y_test.shape)

    # 2) Fazer previsões explicitamente sobre X_test
    y_pred_norm = model.predict(X_test)
    # garantir 1D
    y_pred_norm = np.ravel(y_pred_norm)
    y_test_norm = np.ravel(y_test)

    print("Raw prediction shapes (norm): y_pred_norm =", y_pred_norm.shape, " y_test_norm =", y_test_norm.shape)
    print("Sample preds (norm) first 5:", y_pred_norm[:5])
    print("Sample y_test_norm first 5:", y_test_norm[:5])

    # 3) Segurança: se shapes divergirem, diagnosticar rapidamente
    if y_pred_norm.shape[0] != y_test_norm.shape[0]:
        print("\n!!! ERRO: número de previsões não bate com y_test !!!")
        print("-> Tentando diagnosticar possíveis fontes...")

        # predição sobre treino para ver se o model.predict está retornando len(X_train)
        try:
            y_pred_train_norm = np.ravel(model.predict(X_train))
            print(" predict(X_train).shape:", y_pred_train_norm.shape)
        except Exception as e:
            print(" predict(X_train) falhou:", str(e))

        # mostrar primeiro/último index do conjunto de teste e treino (se tiver index disponível no info)
        print("Tamanhos esperados: n_train={}, n_val={}, n_test={}".format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))
        raise RuntimeError("Previsões com shape inconsistente. Verifique se está prevendo sobre X_test (e se X_test realmente tem o número de amostras esperado).")

    # 4) Desnormalizar previsões e y_test
    def denormalize_target_local(scaler_obj, y_norm_arr, feature_names_list, target_name='Close'):
        idx = feature_names_list.index(target_name)
        # Use n_features consistent with scaler; fall back to len(feature_names_list) se atributo não existir
        n_features_scaler = getattr(scaler_obj, "n_features_in_", len(feature_names_list))
        dummy = np.zeros((len(y_norm_arr), n_features_scaler), dtype=np.float32)
        dummy[:, idx] = y_norm_arr.flatten()
        inv = scaler_obj.inverse_transform(dummy)
        return inv[:, idx]

    # checar scaler
    if not hasattr(scaler, 'n_features_in_'):
        print("⚠️ Aviso: scaler não tem 'n_features_in_'. Usando len(feature_names).")
    else:
        print("scaler.n_features_in_:", scaler.n_features_in_)

    y_pred = denormalize_target_local(scaler, y_pred_norm, feature_names, target_name=target)
    y_true = denormalize_target_local(scaler, y_test_norm, feature_names, target_name=target)

    print("Post-denorm shapes: y_pred =", y_pred.shape, " y_true =", y_true.shape)
    print("Post-denorm sample (pred/true) first 5:")
    for p, t in zip(y_pred[:5], y_true[:5]):
        print(f" pred={p:.4f}  true={t:.4f}")

    # 5) Métricas (desnormalizadas)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # evitar divisão por zero no MAPE
    eps = 1e-9
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape)
    }
    print("\n--- METRICS (desnormalized) ---")
    print(metrics)

    # Salvar artefatos de experimento
    json.dump(metrics, open(RESULTS_DIR / 'metrics.json', 'w'), indent=4)

    # Registrar experimento no CSV
    log_df = pd.DataFrame([{
        'model': 'LSTM',
        'units1': LSTM_UNITS_1,
        'units2': LSTM_UNITS_2,
        'dropout': DROPOUT,
        'batch': BATCH_SIZE,
        'lr': LR,
        'epochs': len(history.history['loss']),
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'notes': NOTES
    }])
    csv_path = RESULTS_DIR / 'experiment_log.csv'
    if csv_path.exists():
        log_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(csv_path, index=False)

    # Salvar histórico de treino
    pickle.dump(history.history, open(RESULTS_DIR / 'history.pkl', 'wb'))

if __name__ == '__main__':
    main()
