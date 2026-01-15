from pathlib import Path

# /opt/render/project/src/api/app
BASE_DIR = Path(__file__).resolve().parent

# /opt/render/project/src/api
API_DIR = BASE_DIR.parent

# Caminhos dos artefatos dentro de api/model
MODEL_PATH = str(API_DIR / "model" / "model.keras")
SCALER_PATH = str(API_DIR / "model" / "scaler.pkl")

# Parâmetros do modelo (definidos no treino)
N_TIMESTEPS = 60
N_FEATURES = 19
