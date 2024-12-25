import mlflow
from mlflow.tracking import MlflowClient

from config import Config

def register_best_model():
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(Config.EXPERIMENT_NAME)
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_auc DESC"]
    )
    
    if runs:
        best_run = runs[0]
        
        model_uri = f"runs:/{best_run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, "employee_access_model")
        
        client.update_model_version(
            name="employee_access_model",
            version=mv.version,
            description="Employee access prediction model using CatBoost"
        )
        
        client.set_model_version_tag(
            name="employee_access_model",
            version=mv.version,
            key="type",
            value="classification"
        )
        
if __name__ == '__main__':
    register_best_model()
