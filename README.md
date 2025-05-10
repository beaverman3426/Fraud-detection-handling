# PaySim Fraud Detection and Asynchronous Prediction API

## Overview
This repository contains a project with two main components:
1. A **Random Forest** model trained on the PaySim dataset to detect fraudulent financial transactions.
2. A **FastAPI** endpoint leveraging **Celery** and **RabbitMQ** to run asynchronous predictions using the trained model.

The project addresses financial fraud detection, a critical task in the financial industry, and provides a scalable API for real-time predictions.

## PaySim Dataset
The **PaySim dataset** is a synthetic financial dataset simulating mobile money transactions, generated to mimic real-world financial activities. It was created to address the scarcity of publicly available financial datasets due to privacy and security concerns.

### Dataset Details
- **Source**: Generated using agent-based simulation by researchers to replicate real transaction patterns.
- **Features**: Includes transaction type (e.g., CASH_IN, CASH_OUT, TRANSFER), amount, customer IDs (`nameOrig`, `nameDest`), and fraud labels (`isFraud`).
- **Target**: Binary classification (`isFraud`: 0 for non-fraudulent, 1 for fraudulent).
- **Size**: Contains millions of transactions, with only **0.13%** labeled as fraudulent, making it highly imbalanced.

### Importance of PaySim
- **Lack of Public Financial Datasets**: Real financial datasets are rarely shared due to sensitive customer information and regulatory restrictions. PaySim provides a realistic alternative for research and model development.
- **Fraud Detection**: Enables testing and benchmarking of fraud detection algorithms in a controlled environment, critical for improving financial security.

## Random Forest Model

### Data Preprocessing: Label Encoding
The dataset is preprocessed to prepare it for the Random Forest model. A key step is encoding the categorical `type` column (e.g., CASH_IN, CASH_OUT) using `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

# Drop irrelevant columns
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Encode the type of transaction
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
```
## Why Label Encoding?

### Suitability for Tree-Based Models
Algorithms like Decision Trees and Random Forests do not assume ordinal relationships in categorical variables. They split nodes based on feature values, so label-encoded integers (e.g., 0 for CASH_IN, 1 for CASH_OUT) work effectively without introducing false assumptions.

### Avoiding One-Hot Encoding
One-hot encoding creates binary columns for each category, increasing dimensionality (e.g., 5 columns for 5 transaction types). Label encoding uses a single column, reducing memory usage and disk space, which is beneficial for large datasets like PaySim.

### Efficiency
Label encoding is computationally lighter and sufficient for Random Forests, as they handle categorical features well without requiring one-hot encoding.

## Handling Imbalanced Data
The PaySim dataset is highly imbalanced, with only **0.13%** of transactions being fraudulent. This poses a challenge for fraud detection, as models may overfit to the majority (non-fraudulent) class.

### SMOTE Consideration
Techniques like Synthetic Minority Oversampling Technique (SMOTE) can generate synthetic fraudulent samples to balance the dataset. However, SMOTE is not used here because:
- Random Forests handle imbalanced data effectively using the `class_weight='balanced'` parameter, which assigns higher weights to the minority class (fraudulent transactions) during training.
- SMOTE may introduce synthetic noise, potentially reducing model generalizability.

### Random Forest Approach
The model uses `class_weight='balanced'` to adjust for the imbalance, ensuring the algorithm focuses on detecting fraudulent transactions without requiring oversampling.

## Model Training
The Random Forest model is trained on the preprocessed PaySim dataset:

### Algorithm
Random Forest is an ensemble of Decision Trees, where each tree is trained on a random subset of data and features. Predictions are aggregated (majority voting for classification) to improve accuracy and robustness.

### Why Random Forest?
- Handles high-dimensional data and non-linear relationships effectively.
- Robust to noise and outliers, common in financial datasets.
- Provides feature importance scores, useful for understanding key predictors of fraud.

## Evaluation Metrics
The model is evaluated using metrics suited for imbalanced classification, each with specific strengths and deficiencies:

### Precision
Measures the proportion of predicted fraudulent transactions that are actually fraudulent (`TP / (TP + FP)`).
- **Use Case**: Precision is critical when false positives (legitimate transactions flagged as fraudulent) are costly, e.g., flagging a customer's transaction may lead to inconvenience or loss of trust. In such cases, we prioritize true positives and minimizing false positives over false negatives.
- **Deficiency**: High precision may come at the cost of missing some fraudulent transactions (low recall).

### Recall
Measures the proportion of actual fraudulent transactions correctly identified (`TP / (TP + FN)`).
- **Use Case**: Recall is vital when false negatives (missing a fraudulent transaction) are more costly, e.g., allowing fraud to go undetected can lead to financial losses. We are more tolerant of false positives but less tolerant of false negatives.
- **Deficiency**: High recall may increase false positives, leading to unnecessary investigations.

### F1-Score
Harmonic mean of precision and recall, balancing both metrics for a comprehensive evaluation.
- **Use Case**: Useful when both false positives and false negatives are important, providing a single metric for model performance.
- **Deficiency**: May not highlight trade-offs between precision and recall in highly imbalanced datasets.

### ROC-AUC
Measures the model's ability to distinguish between classes across all thresholds.
- **Use Case**: Provides an overall performance metric, robust to imbalance.
- **Deficiency**: May be overly optimistic in highly imbalanced datasets like PaySim, where high AUC can mask poor minority class performance.

Given the imbalanced nature of PaySim, **precision**, **recall**, and **F1-score** are prioritized over accuracy, as accuracy can be misleading (e.g., predicting all transactions as non-fraudulent yields 99.87% accuracy but misses all frauds).

## FastAPI Endpoint with Celery and RabbitMQ
The second component is a **FastAPI** endpoint that serves predictions using the trained Random Forest model, integrated with **Celery** and **RabbitMQ** for asynchronous processing.

### FastAPI Setup
FastAPI provides a high-performance API to receive transaction data and return fraud predictions. The endpoint accepts transaction features (e.g., amount, type) and triggers a prediction task.

### Celery and RabbitMQ
#### Celery
A distributed task queue system that handles asynchronous task execution.
- **Role**: Celery offloads the Random Forest prediction task from the main FastAPI thread, allowing the API to remain responsive while the model processes predictions.
- **Why Celery?** It enables scalable, non-blocking task execution, ideal for computationally intensive tasks like ML predictions.

#### RabbitMQ
A message broker that acts as a queue for Celery tasks.
- **Role**: RabbitMQ stores and manages prediction tasks, ensuring reliable communication between FastAPI and Celery workers.
- **Why RabbitMQ?** It provides robust message queuing, fault tolerance, and scalability, ensuring tasks are processed even under high load or server failures.

### How Task Queues Help
- **Offloading ML Predictions**: Running predictions in the main thread can block the API, slowing responses during high traffic. Celery offloads predictions to worker processes, freeing the main thread to handle incoming requests.
- **Asynchronous Processing**: Clients receive an immediate response (e.g., task ID) while predictions are computed in the background, improving user experience.
- **Scalability**: Task queues allow horizontal scaling by adding more Celery workers to handle increased prediction requests, supporting high-traffic scenarios.
- **Reliability**: RabbitMQ ensures tasks are not lost, even if workers crash, by persisting messages until processed.

### Example Workflow
1. A client sends transaction data to the FastAPI endpoint.
2. FastAPI creates a Celery task and sends it to RabbitMQ.
3. A Celery worker retrieves the task, runs the Random Forest model, and stores the prediction.
4. The client queries the API for the prediction result using the task ID.

## Possible Improvements
To enhance the project, consider the following:

### Model Enhancements
- Experiment with other algorithms (e.g., XGBoost, LightGBM) for potentially better performance.
- Use feature engineering to create new features, such as transaction frequency or customer behavior patterns.
- Implement cost-sensitive learning to directly optimize for financial losses from false negatives.

### Data Handling
- Explore advanced imbalance techniques (e.g., ADASYN) if `class_weight='balanced'` is insufficient.
- Validate the model on real-world financial datasets (if available) to ensure generalizability.

### API Scalability
- Deploy the FastAPI application on a containerized platform (e.g., Kubernetes) for better load balancing.
- Optimize Celery worker configurations (e.g., prefetch settings) to handle varying task loads.
- Add caching (e.g., Redis) for frequent predictions to reduce computation time.

### Monitoring and Logging
- Implement monitoring for Celery task queues to detect bottlenecks or failures.
- Add detailed logging for predictions to track model performance and debug issues.

### Security
- Enhance API security with authentication and rate limiting to prevent abuse.
- Encrypt sensitive transaction data in transit and at rest.
