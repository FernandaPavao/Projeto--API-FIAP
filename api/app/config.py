from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # /api/app
ROOT_DIR = BASE_DIR.parent.parent                   # sobe para a raiz do repo

MODEL_PATH = str(ROOT_DIR / "model" / "model.keras")
SCALER_PATH = str(ROOT_DIR / "model" / "scaler.pkl")


# Parâmetros do modelo (definidos no treino)
N_TIMESTEPS = 60
N_FEATURES = 19


