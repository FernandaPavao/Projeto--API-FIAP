from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    data: List[List[float]] = Field(
        ...,
        description=(
            "Lista de N timesteps, onde cada timestep contém "
            "M features numéricas usadas no treino do modelo."
        ),
        example=[
            [10.2, 10.5, 10.1, 10.4, 120000],
            [10.3, 10.6, 10.2, 10.5, 130000]
        ]
    )


class PredictResponse(BaseModel):
    prediction: float = Field(
        ...,
        description="Valor previsto do próximo preço do ativo"
    )
