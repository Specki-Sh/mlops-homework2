# Employee Access Prediction Pipeline
[Amazon.com - Employee Access Challenge](https://www.kaggle.com/competitions/amazon-employee-access-challenge/data)

1. Start MLflow server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

2. Run the main pipeline:
```bash
python main.py
```

3. Register the new model:
```bash
python deployment_mlflow.py
```