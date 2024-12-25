from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

class MetricsCalculator:
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred_proba, prefix=""):
        y_pred = y_pred_proba > 0.5
        
        metrics = {
            f"{prefix}roc_auc": roc_auc_score(y_true, y_pred_proba),
            f"{prefix}precision": precision_score(y_true, y_pred),
            f"{prefix}recall": recall_score(y_true, y_pred),
            f"{prefix}f1": f1_score(y_true, y_pred)
        }
        
        return metrics
