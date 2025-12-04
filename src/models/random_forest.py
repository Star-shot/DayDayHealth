import numpy as np
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
)

class RandomForest:
    """
    Random Forest classifier with evaluation metrics saving and visualization
    
    Parameters:
    n_estimators (int): 森林中树的数量，默认100
    max_depth (int/None): 树的最大深度，默认None（不限）
    min_samples_split (int/float): 分裂节点所需最小样本数，默认2
    min_samples_leaf (int/float): 叶节点最小样本数，默认1
    max_features (str/float): 每棵树的最大特征数，默认'sqrt'
    class_weight (str/dict): 类别权重，默认None
    n_jobs (int): 并行作业数，默认None
    random_state (int): 随机种子，默认42
    out_dir (str): 输出目录，默认'output'
    
    """
    def __init__(self, 
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 class_weight=None,
                 n_jobs=None,
                 random_state=42,
                 out_dir='output'):
        
        # 初始化随机森林参数
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.out_dir = out_dir
        
        # 初始化模型
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
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
        
        # 基础指标
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
        with open(f'{self.out_dir}/rf_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
