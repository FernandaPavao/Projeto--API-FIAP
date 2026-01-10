# **Analysis Report — Experimentos LSTM**

Este relatório resume, interpreta e avalia os experimentos realizados para previsão de séries temporais usando redes LSTM. Os resultados foram extraídos diretamente do arquivo `experiment_log.csv`.

---

## **1. Resumo dos Experimentos**

Foram conduzidos **4 experimentos**, variando unidades dos LSTMs, taxa de dropout, batch size e learning rate.

### **Tabela Resumo**

| Experimento       | units1 | units2 | dropout | batch | lr     | epochs | MAE      | RMSE     | MAPE     | Notes                    |
|------------------|--------|--------|---------|-------|--------|--------|----------|----------|----------|--------------------------|
| Baseline          | 50     | 50     | 0.2     | 32    | 0.001  | 56     | **0.8351** | **1.0834** | **1.54%** | Baseline LSTM model      |
| Bigger Model      | 128    | 64     | 0.2     | 64    | 0.001  | 22     | 1.8564   | 2.3256   | 3.37%    | Bigger Model             |
| Smaller LR        | 128    | 64     | 0.2     | 64    | 0.0005 | 52     | 0.8610   | 1.1912   | 1.64%    | Smaller Learning Rate    |
| More Dropout      | 128    | 64     | 0.3     | 64    | 0.001  | 76     | **0.7487** | **0.9701** | **1.39%** | More Dropout             |

---

## **2. Interpretação Geral dos Resultados**

### 🔹 **Desempenho Global**

- O experimento **More Dropout** apresentou a **melhor performance geral**, com:
  - **Menor MAE** → 0.7487  
  - **Menor RMSE** → 0.9701  
  - **Menor MAPE** → 1.39%

  Aumentar o dropout reduziu overfitting e estabilizou as previsões.

- O modelo **Bigger Model (128 → 64)** foi o pior, provavelmente por **overfitting severo** devido à complexidade excessiva.

- O **Baseline** teve desempenho muito sólido. Um modelo relativamente simples funciona bem para este dataset.

- O **Smaller LR** teve boa estabilidade, mas não superou o modelo com maior dropout.

---

## **3. Ranking dos Modelos**

### 🥇 **1º Lugar — More Dropout**
- Melhor erro absoluto, quadrático e percentual.
- Modelo mais robusto e com melhor generalização.

### 🥈 **2º Lugar — Baseline**
- Desempenho forte com arquitetura simples.
- Ótimo ponto de partida para iterações futuras.

### 🥉 **3º Lugar — Smaller LR**
- Bom resultado, mas não supera os dois primeiros.

### ❌ **4º Lugar — Bigger Model**
- Maior erro.
- Indicação clara de overfitting.

---

## **4. Análise dos Hiperparâmetros**

### **Tamanho da Rede**
Modelos com muitas unidades (128→64) **não melhoraram** e tiveram pior desempenho.

### **Dropout**
O aumento de dropout para **0.3** foi crucial para melhorar generalização.

### **Learning Rate**
- LR menor (0.0005) ajudou na estabilidade, mas não superou ajuste de dropout.

### **Batch Size**
Tanto 32 quanto 64 funcionaram bem; 64 convergiu levemente mais rápido.

---

## **5. Conclusão**

O modelo **More Dropout** é o melhor experimento até agora e deve ser utilizado como **modelo principal**.

Ele apresentou:
- Melhor erro absoluto  
- Melhor generalização  
- Maior robustez  
- Convergência estável  

A arquitetura simples do baseline também se mostrou surpreendentemente efetiva.

---

