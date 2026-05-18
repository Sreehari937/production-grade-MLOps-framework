from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import mlflow.pyfunc

app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.0"
)

model = mlflow.pyfunc.load_model("model")

# Get exact training feature names from sklearn model
sk_model = model._model_impl.sklearn_model
EXPECTED_COLUMNS = list(sk_model.feature_names_in_)


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def home():
    return {
        "message": "Telecom Churn Prediction API running",
        "expected_columns": EXPECTED_COLUMNS
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        data = {col: 0 for col in EXPECTED_COLUMNS}
        data.update(request.features)

        input_df = pd.DataFrame([data])
        input_df = input_df[EXPECTED_COLUMNS]

        prediction = model.predict(input_df)

        return {
            "prediction": int(prediction[0]),
            "churn_status": "Churn" if int(prediction[0]) == 1 else "No Churn"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))