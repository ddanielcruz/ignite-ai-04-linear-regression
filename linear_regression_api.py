import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class RequestBody(BaseModel):
    studied_hours: float


class ResponseBody(BaseModel):
    score: float


# Load the model
model = joblib.load("models/linear_regression_model.pkl")


@app.post("/predict", response_model=ResponseBody)
def predict_score(request: RequestBody):
    studied_hours = request.studied_hours
    score = model.predict([[studied_hours]])[0].round()

    return {"score": score}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
