"""
K-Nearest Neighbors 分类器
"""
import numpy as np
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
)


class KNN:
    """
    K-Nearest Neighbors 分类器
    
    参数：
    n_neighbors (int): 近邻数量，默认5
    weights (str): 权重方式，'uniform'或'distance'，默认'uniform'
    algorithm (str): 算法，'auto', 'ball_tree', 'kd_tree', 'brute'，默认'auto'
    metric (str): 距离度量，默认'minkowski'
    out_dir (str): 输出目录
    """
    
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 metric='minkowski',
                 out_dir='output'):
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.out_dir = out_dir
        
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric
        )
        self._create_output_dir()
        self.classes_ = None
    
    def _create_output_dir(self):
        os.makedirs(f'{self.out_dir}', exist_ok=True)
    
    def train(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True, zero_division=0)
        }
        
        try:
            if len(self.classes_) == 2:
                metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics['auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr')
                metrics['auc_ovo'] = roc_auc_score(y, y_proba, multi_class='ovo')
        except Exception as e:
            print(f"AUC计算错误: {str(e)}")
            metrics['auc'] = None
        
        self._save_metrics(metrics)
        return metrics
    
    def _save_metrics(self, metrics):
        with open(f'{self.out_dir}/knn_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

