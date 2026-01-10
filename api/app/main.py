from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse
from .services import predict_next_value


app = FastAPI(
    title="LSTM Stock Price Prediction API",
    description=(
        "API RESTful para previsão do próximo valor de um ativo financeiro "
        "utilizando um modelo LSTM previamente treinado."
    ),
    version="1.0.0"
)


@app.get("/")
def health_check():
    return {"status": "API online"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        prediction = predict_next_value(request.data)
        return {"prediction": prediction}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Erro interno durante a predição"
        )

