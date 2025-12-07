"""
XGBoost 分类器
"""
import numpy as np
import json
import os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: xgboost 未安装，请运行 pip install xgboost")

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
)


class XGBoost:
    """
    XGBoost 分类器
    
    参数：
    n_estimators (int): 树的数量，默认100
    max_depth (int): 树的最大深度，默认6
    learning_rate (float): 学习率，默认0.1
    subsample (float): 样本采样比例，默认1.0
    colsample_bytree (float): 特征采样比例，默认1.0
    random_state (int): 随机种子
    out_dir (str): 输出目录
    """
    
    def __init__(self,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 random_state=None,
                 out_dir='output'):
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost 未安装，请运行 pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.out_dir = out_dir
        
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
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
        with open(f'{self.out_dir}/xgboost_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

