import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
)

class LogisticRegression:
    """
    逻辑回归分类器（支持多分类）包含评估指标保存功能
    
    参数：
    penalty (str): 正则化类型，默认'l2'，可选'l1', 'l2', 'elasticnet', 'none'
    C (float): 正则化强度的倒数，默认1.0（值越小正则化越强）
    solver (str): 优化算法，默认'lbfgs'，可选'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    max_iter (int): 最大迭代次数，默认100
    class_weight (str/dict): 类别权重，默认None，可选'balanced'
    multi_class (str): 多分类策略，默认'auto'，可选'ovr', 'multinomial'
    random_state (int): 随机种子，默认None
    out_dir (str): 输出目录，默认'output'
    """
    
    def __init__(self, 
                 penalty='l2',
                 C=1.0,
                 solver='lbfgs',
                 max_iter=100,
                 class_weight=None,
                 multi_class='auto',
                 random_state=None,
                 out_dir='output'):
        
        # 初始化参数
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.random_state = random_state
        self.out_dir = out_dir
        
        # 初始化模型
        self.model = SklearnLogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            multi_class=self.multi_class,
            random_state=self.random_state
        )
        self._create_output_dir()
        self.classes_ = None

    def _create_output_dir(self):
        """创建输出目录"""
        os.makedirs(f'{self.out_dir}', exist_ok=True)

    def train(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        self.classes_ = self.model.classes_  # 记录类别标签

    def predict(self, X):
        """生成预测结果"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """生成概率估计"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """计算评估指标"""
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

        # 多分类AUC计算
        try:
            if len(self.classes_) == 2:
                metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics['auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr')
                metrics['auc_ovo'] = roc_auc_score(y, y_proba, multi_class='ovo')
        except Exception as e:
            print(f"AUC计算错误: {str(e)}")
            metrics['auc'] = None
        
        # 保存指标
        self._save_metrics(metrics)
        return metrics

    def _save_metrics(self, metrics):
        """保存指标到JSON文件"""
        with open(f'{self.out_dir}/logistic_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

