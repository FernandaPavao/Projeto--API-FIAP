from pathlib import Path

# Pasta API (um nível acima de Aplicativo)
BASE_DIR = Path(__file__).resolve().parent.parent

# Caminhos dos artefatos de modelo (ajuste os nomes se forem diferentes)
MODEL_PATH = BASE_DIR / "Modelo" / "model.keras"
SCALER_PATH = BASE_DIR / "Modelo" / "scaler.pkl"

# Parâmetros do modelo
N_TIMESTEPS = 60
N_FEATURES = 19
