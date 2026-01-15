# Tech Challenge Fase 4 - Previsão de Ações com LSTM

**Disciplina:** Machine Learning Engineering  
**Projeto:** Previsão de preços de ações usando Deep Learning (LSTM)  
**Empresa Analisada:** Vale S.A. (VALE3.SA)

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Descrição dos Scripts](#descrição-dos-scripts)
- [Dados Gerados](#dados-gerados)
- [Visualizações](#visualizações)
- [Próximos Passos](#próximos-passos)
- [Treinamento do Modelo LSTM](#treinamento-do-modelo-lstm)
- [Experimentos Realizados](#experimentos-realizados)
- [Artefatos Gerados](#artefatos-gerados)
- [Scripts Complementares](#scripts-complementares)
- [Relatórios e Análises](#relatórios-e-análises)

---

## 🎯 Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning para previsão de preços de ações utilizando redes neurais LSTM (Long Short-Term Memory). 

O trabalho envolve:

1. Coleta e pré-processamento de dados
2. Preparação dos dados para modelos sequenciais
3. Desenvolvimento, treinamento e avaliação de modelos LSTM
4. Geração de artefatos de modelo, experimentos e relatórios


### Características do Dataset

- **Ativo:** VALE3.SA (Vale do Rio Doce)
- **Período:** Últimos 5 anos
- **Frequência:** Diária
- **Features:** 
  - Básicas: Open, High, Low, Close, Volume
  - Indicadores Técnicos: SMA, EMA, RSI, MACD, Bollinger Bands, Volatilidade
- **Janela Temporal:** 60 dias (lookback)
- **Divisão:** 70% treino / 15% validação / 15% teste
  ----
## API em Produção

- URL base: https://projeto-api-fiap-xqxb.onrender.com
- Documentação (Swagger): https://projeto-api-fiap-xqxb.onrender.com/docs

### Endpoint de previsão

**POST** `/predict`

**Request (JSON)**

```json
{
  "data": [
    [ ... 19 números ... ],
    ...
  ]
}

---

## 📁 Estrutura do Projeto

```
tech_challenge_fase4/
│
├── api/                   
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── services.py
│   ├── model/
│   │   ├── model.keras
│   │   └── scaler.pkl
│   ├── requirements.txt
│   └── Dockerfile

├── data/
│ ├── raw/ ← Dados Brutos do yFinance
│ │ └── VALE3_SA_raw.csv
│ └── processed/ ← Dados Processados e prontos para treino
│ ├── VALE3_SA_processed.csv
│ ├── scaler.pkl ← Scaler para normalização
│ ├── train_data.npz ← Dados de treino (X, y)
│ ├── val_data.npz ← Dados de validação (X, y)
│ ├── test_data.npz ← Dados de teste (X, y)
│ └── data_info.json ← Metadados do dataset
│
├── src/
│ ├── config.py  ← Configuração do projeto
│ ├── data_collection.py ← Script de coleta de dados
│ ├── data_preprocessing.py ← Script de pré-processamento
│ ├── data_preparation.py ← Script de preparação para LSTM
│ ├── train_model.py ← Treinamento do modelo LSTM
│ ├── test_model.py ← Avaliação do modelo
│ ├── eda_analysis.py ← Análise exploratória dos dados
│ └── plot_experiments.py ← Visualização de experimentos
│ 
│
├── models/ 
│ └──  best_model.keras ← Melhor modelo
├── results/
│ ├── metrics.json
│ ├── metrics_test.json
│ ├── predictions.csv
│ ├── pred_vs_real.png
│ ├── error_distribution.png
│ ├── experiment_log.csv
│ ├── experiment_rank_plot.png
│ ├── experiment_summary.png
│ ├── history.pkl
│ └── analysis_report.md
│
├── reports/
│ └── figures/
│ ├── 01_price_history.png
│ ├── 02_returns_distribution.png
│ ├── 03_moving_averages.png
│ ├── 04_technical_indicators.png
│ ├── 05_correlation_matrix.png
│ └── summary_statistics.txt
│
├── requirements.txt
├── README.md
└── run_pipeline.py
```

---

## 🔧 Requisitos

- Python 3.8 ou superior
- Bibliotecas listadas em `requirements.txt`

---

## 📦 Instalação

### 1. Clone ou baixe o projeto

```bash
git clone <seu-repositorio>
cd tech_challenge_fase4
```

### 2. Crie um ambiente virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### Opção 1: Executar todo o pipeline de uma vez

```bash
python run_pipeline.py
```

Este script executa automaticamente:
1. Coleta de dados
2. Pré-processamento
3. Preparação para LSTM
4. Análise exploratória (EDA)

### Opção 2: Executar scripts individualmente

#### Passo 1: Coletar dados

```bash
python src/data_collection.py
```

**O que faz:**
- Baixa dados históricos da VALE3.SA usando yfinance
- Valida os dados coletados
- Salva em `data/raw/VALE3_SA_raw.csv`

#### Passo 2: Pré-processar dados

```bash
python src/data_preprocessing.py
```

**O que faz:**
- Remove valores nulos
- Detecta e trata outliers
- Adiciona indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger Bands)
- Salva em `data/processed/VALE3_SA_processed.csv`

#### Passo 3: Preparar dados para LSTM

```bash
python src/data_preparation.py
```

**O que faz:**
- Normaliza os dados com MinMaxScaler
- Cria sequências temporais (janelas de 60 dias)
- Divide em treino/validação/teste (70/15/15)
- Salva arquivos `.npz` prontos para treino

#### Passo 4: Análise Exploratória (Opcional)

```bash
python src/eda_analysis.py
```

**O que faz:**
- Gera gráficos de análise
- Cria estatísticas descritivas
- Salva visualizações em `reports/figures/`

#### Passo 5: Treinamento do modelo LSTM

```bash
python ./src/train_model.py
```
#### Passo 6: Teste e avaliação do modelo

```bash
python ./src/test_model.py
```

#### Passo 7: Plot dos experimentos

```bash
python ./src/plot_experiments.py
```
---

## 🚀 API de Previsão com LSTM 

Esta seção descreve a camada de serving e deploy do modelo LSTM treinado,
responsável por disponibilizar o modelo como um serviço de API RESTful.

A API foi desenvolvida seguindo boas práticas de Machine Learning Engineering,
com separação clara entre a etapa de treinamento do modelo e a etapa de inferência
em produção.

### 🎯 Objetivo da API

- Servir o modelo LSTM treinado via API
- Receber janelas temporais de séries históricas
- Validar formato e dimensionalidade dos dados
- Aplicar o scaler utilizado no treinamento
- Retornar a previsão do próximo valor do ativo

### 🧱 Arquitetura da API

- **Framework:** FastAPI
- **Modelo:** LSTM treinado (Keras `.keras`)
- **Scaler:** MinMaxScaler (`.pkl`)
- **Entrada:** 60 timesteps × 19 features
- **Saída:** Previsão do próximo valor do preço

### 📁 Localização no Projeto

A API está localizada na pasta `api/` do projeto, mantendo separação clara entre
treinamento do modelo e inferência em produção.

### ▶️ Execução da API

```bash
cd api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8001

http://127.0.0.1:8001/docs




### 🐳 Docker

O projeto inclui um Dockerfile preparado para containerização da API.
A execução via Docker depende de Docker Desktop, WSL 2 e virtualização ativa.
Em ambientes corporativos, essa execução pode ser restrita.



## 📊 Descrição dos Scripts

### `config.py`
Centraliza todas as configurações do projeto:
- Parâmetros da coleta de dados
- Configurações do modelo (lookback, features)
- Divisão dos dados
- Caminhos de arquivos

### `data_collection.py`
Responsável pela coleta de dados:
- Usa biblioteca `yfinance` para baixar dados históricos
- Valida integridade dos dados
- Detecta valores nulos, negativos e duplicados

### `data_preprocessing.py`
Realiza limpeza e engenharia de features:
- Tratamento de valores ausentes (interpolação)
- Remoção/tratamento de outliers (método IQR)
- Adição de 15+ indicadores técnicos
- Validação final dos dados

### `data_preparation.py`
Prepara dados especificamente para LSTM:
- Normalização com MinMaxScaler [0, 1]
- Criação de sequências temporais (janelas deslizantes)
- Divisão temporal em treino/validação/teste
- Salva dados em formato otimizado (`.npz`)

### `eda_analysis.py`
Análise exploratória completa:
- Histórico de preços e volume
- Distribuição de retornos
- Médias móveis e indicadores técnicos
- Matriz de correlação
- Estatísticas descritivas

### `train_model.py`
Treina e salva o modelo:
- Carrega dados e scaler
- Cria e treina o modelo LSTM
- Usa callbacks: EarlyStopping, ReduceLROnPlateau e ModelCheckpoint
- Salva:
   - best_model.keras
   - history.pkl
   - metrics.json
   - Registro no experiment_log.csv

### `test_model.py`
Testa e salva as métricas e imagens:
- Carrega o modelo salvo.
- Realiza previsões no conjunto de teste.
- Desnormaliza a saída.
- Gera:
   - predictions.csv
   - pred_vs_real.png
   - error_distribution.png
   - metrics_test.json

### `plot_experiments.py`
Compara os experimentos e rankeia eles:
- Lê o experiments_log.csv
- Plota ranking dos modelos
- Gera:
   - experiment_rank_plot.png
   - experiment_summary.png

---

## 💾 Dados Gerados

Os seguintes arquivos estão prontos para uso no treinamento do modelo LSTM:

1. **`train_data.npz`**
   - `X_train`: shape (n_samples, 60, n_features)
   - `y_train`: shape (n_samples,)

2. **`val_data.npz`**
   - `X_val`: shape (n_samples, 60, n_features)
   - `y_val`: shape (n_samples,)

3. **`test_data.npz`**
   - `X_test`: shape (n_samples, 60, n_features)
   - `y_test`: shape (n_samples,)

4. **`scaler.pkl`**
   - Objeto MinMaxScaler salvo com pickle
   - Necessário para desnormalizar as previsões

5. **`data_info.json`**
   - Metadados do dataset (features, shapes, datas, etc.)

### Como carregar os dados

```python
import numpy as np
import pickle

# Carregar dados de treino
train_data = np.load('data/processed/train_data.npz')
X_train = train_data['X_train']
y_train = train_data['y_train']

# Carregar dados de validação
val_data = np.load('data/processed/val_data.npz')
X_val = val_data['X_val']
y_val = val_data['y_val']

# Carregar dados de teste
test_data = np.load('data/processed/test_data.npz')
X_test = test_data['X_test']
y_test = test_data['y_test']

# Carregar scaler (para desnormalizar previsões)
with open('data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Exemplo de desnormalização
# predictions_original = scaler.inverse_transform(predictions_normalized)
```

---

## 📈 Visualizações

Após executar `eda_analysis.py`, os seguintes gráficos são gerados:

1. **01_price_history.png**: Histórico de preços e volume
2. **02_returns_distribution.png**: Distribuição de retornos diários
3. **03_moving_averages.png**: Preço com médias móveis
4. **04_technical_indicators.png**: RSI, MACD e Bollinger Bands
5. **05_correlation_matrix.png**: Correlação entre features
6. **summary_statistics.txt**: Estatísticas descritivas completas

---

## 📝 Configurações Importantes

### Modificar parâmetros

Edite o arquivo `src/config.py` para alterar:

```python
# Ativo e período
STOCK_SYMBOL = "VALE3.SA"
START_DATE = '2019-10-17'  # Automático: últimos 5 anos
END_DATE = '2024-10-17'    # Automático: hoje

# Janela temporal
LOOKBACK_DAYS = 60  # Dias para "olhar" para trás

# Divisão dos dados
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Indicadores técnicos
ADD_TECHNICAL_INDICATORS = True
SMA_PERIODS = [7, 21, 50]
RSI_PERIOD = 14
```

---

### Arquitetura utilizada

Modelo LSTM Dropout maior

```python
LSTM(128, return_sequences=True)
Dropout(0.3)
BatchNormalization()
LSTM(64)
Dropout(0.3)
Dense(32, activation='relu')
Dense(1)

```
Parâmetros do treinamento

- Optimizer: Adam
- Loss: MSE
- Batch size: 32–64
- Epochs: até 100
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

Métricas utilizadas

- MAE
- RMSE
- MAPE

### Experimentos realizados

Os experimentos foram executados e documentados em:
   - results/experiment_log.csv
   - results/experiments_rank_lot.png
   - results/experiment_summary.png
   - results/analysis_report.md

## 🐛 Troubleshooting

### Erro: "No module named 'yfinance'"
```bash
pip install yfinance
```

### Erro: "FileNotFoundError"
Execute os scripts na ordem correta:
1. `data_collection.py`
2. `data_preprocessing.py`
3. `data_preparation.py`

### Erro de data no yfinance
Verifique sua conexão com a internet. O yfinance precisa acessar o Yahoo Finance.

---

## 📚 Referências

- **yfinance**: https://pypi.org/project/yfinance/
- **LSTM**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **MinMaxScaler**: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- **Indicadores Técnicos**: https://github.com/bukosabino/ta

---

## ✅ Checklist de Entrega - Pessoa 1

- [x] Coleta de dados históricos (yfinance)
- [x] Tratamento de valores nulos
- [x] Detecção e tratamento de outliers
- [x] Normalização com MinMaxScaler
- [x] Adição de indicadores técnicos
- [x] Criação de janelas temporais (60 dias)
- [x] Divisão treino/validação/teste (70/15/15)
- [x] Salvamento dos dados processados
- [x] Documentação completa
- [x] Análise exploratória (EDA)
- [x] Scripts organizados e comentados

---
## ✅ Checklist de Entrega - Pessoa 3
- [x] Carregamento do modelo LSTM treinado (.keras)
- [x] Carregamento do scaler utilizado no treinamento (.pkl)
- [x] Implementação de API RESTful com FastAPI
- [x] Validação da entrada (timesteps e número de features)
- [x] Pré-processamento para inferência (scaling e reshape)
- [x] Inferência do modelo e desnormalização da previsão
- [x] Tratamento de erros e respostas HTTP apropriadas
- [x] Documentação do endpoint `/predict`
- [x] Estruturação da aplicação em camada de API (`api/`)
- [x] Criação de `requirements.txt` para inferência
- [x] Criação de `Dockerfile` para containerização


## 👥 Autores

**Pessoa 1:** Coleta e Pré-processamento dos Dados ✅  
**Pessoa 2:** Desenvolvimento do Modelo LSTM  ✅
**Pessoa 3:** Deploy da API  ✅
**Pessoa 4:** Produção e Monitoramento  

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do Tech Challenge da Fase 4.

---


**Última atualização:** Dezembro 2025
