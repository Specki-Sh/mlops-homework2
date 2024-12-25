import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import mlflow

class ModelOptimizer:
    def __init__(self, config):
        self.config = config
        
    def objective(self, trial, X_train, X_val, y_train, y_val):
        """Optimization objective for Optuna"""
        with mlflow.start_run(nested=True):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
                'random_seed': self.config.RANDOM_STATE
            }
            
            mlflow.log_params(params)
            
            model = CatBoostClassifier(**params, verbose=0)
            model.fit(X_train, y_train, cat_features=self.config.CAT_FEATURES)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_pred)
            
            mlflow.log_metric("val_auc", auc_score)
            
            return auc_score
    
    def optimize(self, X_train, X_val, y_train, y_val):
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
            n_trials=self.config.N_TRIALS
        )
        return study