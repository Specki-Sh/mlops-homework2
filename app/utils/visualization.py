import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import mlflow

class Visualizer:
    @staticmethod
    def plot_roc_curve(y_true, y_pred):
        """Plot ROC curve and save to MLflow"""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_pred):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()
    
    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Plot feature importance and save to MLflow"""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Feature Importance')
        
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        return importance
