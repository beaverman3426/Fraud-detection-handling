from fastapi import FastAPI
from pydantic import BaseModel
from tasks import predict_task

app = FastAPI()

class Transaction(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.post("/predict/")
def predict(transaction: Transaction):
    task = predict_task.delay(transaction.model_dump())
    return {"task_id": task.id, "status": "processing"} 