from celery import Celery
from predict import predict_fraud

celery = Celery(
    'app',  
    broker='amqp://localhost',
    backend='rpc://',
    include=['tasks'],
    worker_pool_restarts=True 
)

@celery.task
def predict_task(input_data):
    return predict_fraud(input_data)
