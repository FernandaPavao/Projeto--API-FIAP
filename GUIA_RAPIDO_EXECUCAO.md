# 🚀 Guia Rápido de Execução - Pessoa 1

## ⚡ Execução Rápida (Recomendado)

### 1️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Executar pipeline completo
```bash
python run_pipeline.py
```

✅ **Pronto!** Todo o trabalho da Pessoa 1 será executado automaticamente.

---

## 📂 O que será gerado?

### Dados para a Pessoa 2 (em `data/processed/`)
- ✅ `train_data.npz` - Dados de treino
- ✅ `val_data.npz` - Dados de validação  
- ✅ `test_data.npz` - Dados de teste
- ✅ `scaler.pkl` - Scaler para desnormalização
- ✅ `data_info.json` - Metadados

### Visualizações (em `reports/figures/`)
- 📊 5 gráficos de análise
- 📄 Arquivo com estatísticas

---

## 🔧 Execução Passo a Passo (Alternativa)

Se preferir executar cada etapa manualmente:

```bash
# Passo 1: Coletar dados
python src/data_collection.py

# Passo 2: Pré-processar
python src/data_preprocessing.py

# Passo 3: Preparar para LSTM
python src/data_preparation.py

# Passo 4: Análise exploratória (opcional)
python src/eda_analysis.py
```

---

## 📊 Como verificar os resultados?

### 1. Verificar arquivos gerados
```bash
# Windows
dir data\processed

# Linux/Mac
ls -lh data/processed/
```

### 2. Ver informações do dataset
```bash
# Windows
type data\processed\data_info.json

# Linux/Mac
cat data/processed/data_info.json
```

### 3. Ver gráficos
Abra a pasta `reports/figures/` e visualize os arquivos `.png`

---

## 🎯 Entregar para a Pessoa 2

### Arquivos obrigatórios:
1. Pasta `data/processed/` completa
2. Arquivo `src/config.py` (com as configurações)
3. Arquivo `README.md` (documentação)

### Instruções para a Pessoa 2:

```python
# Como carregar os dados preparados
import numpy as np
import pickle
import json

# Carregar treino
train = np.load('data/processed/train_data.npz')
X_train = train['X_train']  # Shape: (samples, 60, features)
y_train = train['y_train']  # Shape: (samples,)

# Carregar validação
val = np.load('data/processed/val_data.npz')
X_val = val['X_val']
y_val = val['y_val']

# Carregar teste
test = np.load('data/processed/test_data.npz')
X_test = test['X_test']
y_test = test['y_test']

# Carregar scaler (importante para desnormalizar depois!)
with open('data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Carregar info
with open('data/processed/data_info.json', 'r') as f:
    info = json.load(f)
    
print(f"Features: {info['dataset_info']['features']}")
print(f"Treino: {X_train.shape}")
```

---

## ⚠️ Problemas Comuns

### Erro: "No module named 'yfinance'"
**Solução:**
```bash
pip install yfinance
```

### Erro: "Permission denied" ao salvar arquivos
**Solução:**  
Execute o script com permissões adequadas ou verifique se as pastas `data/` e `reports/` existem.

### Dados não foram coletados
**Solução:**  
Verifique sua conexão com a internet. O yfinance precisa acessar o Yahoo Finance.

### Quero usar outra empresa
**Solução:**  
Edite `src/config.py` e mude:
```python
STOCK_SYMBOL = "PETR4.SA"  # Exemplo: Petrobras
```

---

## 📋 Checklist Final

Antes de entregar para a Pessoa 2:

- [ ] Pipeline executado sem erros
- [ ] Todos os arquivos em `data/processed/` foram gerados
- [ ] Gráficos em `reports/figures/` estão OK
- [ ] `README.md` está na raiz do projeto
- [ ] Testei carregar os dados `.npz` (código acima)

---

## 📞 Suporte

Se tiver dúvidas, verifique:
1. `README.md` - Documentação completa
2. Mensagens de erro no terminal
3. Arquivo `data/processed/data_info.json` para debug

---

**Boa sorte! 🚀**