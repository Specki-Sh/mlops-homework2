from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
import os
import pandas as pd
from mlflow.models.signature import infer_signature
from utils.metrics import MetricsCalculator

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.artifacts_path = os.path.join(os.getcwd(), 'artifacts')
        os.makedirs(self.artifacts_path, exist_ok=True)
        
    def train_model(self, X_train, X_val, y_train, y_val, params):
        model = CatBoostClassifier(**params, verbose=100)
        
        mlflow.log_params(params)
        
        model.fit(
            X_train,
            y_train,
            cat_features=self.config.CAT_FEATURES,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
        
        train_metrics = MetricsCalculator.calculate_classification_metrics(
            y_train, train_preds, prefix="train_"
        )
        val_metrics = MetricsCalculator.calculate_classification_metrics(
            y_val, val_preds, prefix="val_"
        )
        
        metrics = {
            **train_metrics,
            **val_metrics,
            "best_iteration": model.get_best_iteration()
        }
        
        mlflow.log_metrics(metrics)
        
        self._save_plots(model)
        
        self._log_model_with_signature(model, X_train, y_train)
        
        model_path = os.path.join(self.artifacts_path, 'model.cbm')
        model.save_model(model_path)
        mlflow.log_artifact(model_path)
        
        return model
    
    def _log_model_with_signature(self, model, X, y):
        signature = infer_signature(X, model.predict_proba(X))
        
        input_example = X.head(1)
        
        mlflow.catboost.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example,
            registered_model_name="employee_access_model"
        )
        
        example_path = os.path.join(self.artifacts_path, 'input_example.csv')
        input_example.to_csv(example_path, index=False)
        mlflow.log_artifact(example_path)
    
    def _save_plots(self, model):
        """Save model plots to artifacts directory"""
        try:
            loss_plot_path = os.path.join(self.artifacts_path, "training_loss.png")
            model.plot_metrics().figure_.savefig(loss_plot_path)
            mlflow.log_artifact(loss_plot_path)
            
            importance_plot_path = os.path.join(self.artifacts_path, "feature_importance.png")
            model.plot_feature_importance().figure_.savefig(importance_plot_path)
            mlflow.log_artifact(importance_plot_path)
            
        except Exception as e:
            error_path = os.path.join(self.artifacts_path, 'plot_errors.txt')
            with open(error_path, 'w') as f:
                f.write(str(e))
            mlflow.log_artifact(error_path)