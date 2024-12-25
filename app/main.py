import mlflow
import pandas as pd
from config import Config
from preprocessing.data_loader import DataLoader
from preprocessing.feature_processor import FeatureProcessor
from model.optimizer import ModelOptimizer
from model.trainer import ModelTrainer
from utils.visualization import Visualizer

def main():
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="employee_access_pipeline"):
        # Load and prepare data
        data_loader = DataLoader(Config)
        feature_processor = FeatureProcessor(Config)
        
        train_df, test_df = data_loader.load_data()
        
        X = feature_processor.prepare_features(train_df)
        y = feature_processor.prepare_target(train_df)
        
        X_train, X_val, y_train, y_val = data_loader.split_data(X, y)
        
        optimizer = ModelOptimizer(Config)
        study = optimizer.optimize(X_train, X_val, y_train, y_val)
        
        mlflow.log_params(study.best_params)
        
        trainer = ModelTrainer(Config)
        model = trainer.train_model(X_train, X_val, y_train, y_val, study.best_params)
        
        val_predictions = model.predict_proba(X_val)[:, 1]
        test_predictions = model.predict_proba(feature_processor.prepare_features(test_df))[:, 1]
        
        visualizer = Visualizer()
        visualizer.plot_roc_curve(y_val, val_predictions)
        visualizer.plot_feature_importance(model, Config.CAT_FEATURES)
        
        mlflow.catboost.log_model(model, "model")
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'ACTION': test_predictions
        })
        submission.to_csv('submission.csv', index=False)
        mlflow.log_artifact('submission.csv')

if __name__ == "__main__":
    main()