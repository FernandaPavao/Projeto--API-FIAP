# Caminhos dos artefatos (a partir da raiz do projeto)
from pathlib import Path

# Caminhos dos artefatos (relativos a este arquivo)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model" / "model.keras")
SCALER_PATH = str(BASE_DIR / "model" / "scaler.pkl")


# Parâmetros do modelo (definidos no treino)
N_TIMESTEPS = 60
N_FEATURES = 19

