"""
Naive Bayes 分类器
"""
import numpy as np
import json
import os
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
)


class NaiveBayes:
    """
    Naive Bayes 分类器
    
    参数：
    nb_type (str): 朴素贝叶斯类型，'gaussian', 'multinomial', 'bernoulli'，默认'gaussian'
    var_smoothing (float): 方差平滑参数（仅Gaussian），默认1e-9
    alpha (float): 拉普拉斯平滑参数（Multinomial/Bernoulli），默认1.0
    out_dir (str): 输出目录
    """
    
    def __init__(self,
                 nb_type='gaussian',
                 var_smoothing=1e-9,
                 alpha=1.0,
                 out_dir='output'):
        
        self.nb_type = nb_type
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.out_dir = out_dir
        
        if nb_type == 'gaussian':
            self.model = GaussianNB(var_smoothing=self.var_smoothing)
        elif nb_type == 'multinomial':
            self.model = MultinomialNB(alpha=self.alpha)
        elif nb_type == 'bernoulli':
            self.model = BernoulliNB(alpha=self.alpha)
        else:
            raise ValueError(f"不支持的朴素贝叶斯类型: {nb_type}")
        
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
        with open(f'{self.out_dir}/naive_bayes_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

