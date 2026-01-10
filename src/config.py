"""
Configurações do Projeto - Tech Challenge Fase 4
Centralizador de parâmetros e constantes
"""

from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# CONFIGURAÇÕES DA EMPRESA E PERÍODO
# ============================================================================
STOCK_SYMBOL = "VALE3.SA"  # Vale S.A. (B3)
START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 anos atrás
END_DATE = datetime.now().strftime('%Y-%m-%d')  # Hoje

# ============================================================================
# CONFIGURAÇÕES DO MODELO
# ============================================================================
LOOKBACK_DAYS = 60  # Janela temporal: últimos 60 dias
FORECAST_HORIZON = 1  # Prever 1 dia à frente
TARGET = 'Close'  # Variável alvo: preço de fechamento

# ============================================================================
# DIVISÃO DOS DADOS
# ============================================================================
TRAIN_SIZE = 0.70  # 70% treino
VAL_SIZE = 0.20    # 20% validação
TEST_SIZE = 0.10   # 10% teste

# ============================================================================
# PRÉ-PROCESSAMENTO
# ============================================================================
# Normalização
FEATURE_RANGE = (0, 1)  # MinMaxScaler range
RANDOM_STATE = 42

# Indicadores Técnicos
ADD_TECHNICAL_INDICATORS = True
SMA_PERIODS = [7, 21, 50]  # Médias Móveis Simples
EMA_PERIODS = [12, 26]     # Médias Móveis Exponenciais
RSI_PERIOD = 14            # Relative Strength Index
MACD_PARAMS = {
    'fast': 12,
    'slow': 26,
    'signal': 9
}
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Tratamento de Outliers
OUTLIER_METHOD = 'IQR'  # Método IQR (Interquartile Range)
OUTLIER_THRESHOLD = 1.5  # Multiplicador do IQR

# ============================================================================
# CAMINHOS DOS ARQUIVOS
# ============================================================================
# Diretórios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Criar diretórios se não existirem
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Arquivos de dados
RAW_DATA_FILE = RAW_DATA_DIR / f'{STOCK_SYMBOL.replace(".", "_")}_raw.csv'
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / f'{STOCK_SYMBOL.replace(".", "_")}_processed.csv'

# Arquivos para LSTM
SCALER_FILE = PROCESSED_DATA_DIR / 'scaler.pkl'
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / 'train_data.npz'
VAL_DATA_FILE = PROCESSED_DATA_DIR / 'val_data.npz'
TEST_DATA_FILE = PROCESSED_DATA_DIR / 'test_data.npz'
DATA_INFO_FILE = PROCESSED_DATA_DIR / 'data_info.json'

# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ============================================================================
FIGURE_SIZE = (14, 6)
DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# MENSAGENS E LOGS
# ============================================================================
VERBOSE = True  # Mostrar mensagens detalhadas

def print_config():
    """Imprime as configurações do projeto"""
    print("\n" + "="*60)
    print("CONFIGURAÇÕES DO PROJETO")
    print("="*60)
    print(f"📊 Empresa: {STOCK_SYMBOL}")
    print(f"📅 Período: {START_DATE} até {END_DATE}")
    print(f"🔍 Lookback: {LOOKBACK_DAYS} dias")
    print(f"🎯 Target: {TARGET}")
    print(f"📂 Divisão: {TRAIN_SIZE*100:.0f}% treino / {VAL_SIZE*100:.0f}% val / {TEST_SIZE*100:.0f}% teste")
    print(f"📈 Indicadores Técnicos: {'Sim' if ADD_TECHNICAL_INDICATORS else 'Não'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_config()