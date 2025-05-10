from celery import Celery
from predict import predict_fraud

celery = Celery("tasks",broker="amqp://localhost")

@celery.task
def predict_task(input_data):
    return predict_fraud(input_data)
