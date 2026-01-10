import numpy as np
import pickle
from tensorflow.keras.models import load_model

from .config import MODEL_PATH, SCALER_PATH, N_TIMESTEPS, N_FEATURES


# =========================
# Carregamento dos artefatos
# =========================

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


# =========================
# Função de predição
# =========================

def predict_next_value(data: list) -> float:
    """
    Recebe uma lista com shape (N_TIMESTEPS, N_FEATURES)
    e retorna a previsão do próximo valor na escala original.
    """

    # Converter para numpy array
    data_array = np.array(data)

    # Validar shape
    if data_array.shape != (N_TIMESTEPS, N_FEATURES):
        raise ValueError(
            f"Shape inválido. Esperado ({N_TIMESTEPS}, {N_FEATURES}), "
            f"recebido {data_array.shape}"
        )

    # Normalização
    data_scaled = scaler.transform(data_array)

    # Ajustar shape para LSTM: (1, timesteps, features)
    X = data_scaled.reshape(1, N_TIMESTEPS, N_FEATURES)

    # Previsão
    prediction_scaled = model.predict(X)

    # Desnormalização
    dummy = np.zeros((1, N_FEATURES))
    dummy[0, 0] = prediction_scaled[0, 0]

    prediction = scaler.inverse_transform(dummy)[0, 0]

    return float(prediction)

