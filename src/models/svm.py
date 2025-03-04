import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

class SVM:
    """
    SVM分类器，支持评估指标保存和可视化
    
    参数:
    kernel (str): 核函数类型，默认为'linear'
    C (float): 正则化参数，默认为1.0
    gamma (str/float): 核函数系数，默认为'scale'
    """
    def __init__(self, kernel='linear', C=1.0, gamma='scale', out_dir='output'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.out_dir = out_dir
        self.model = svm.SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,  # 启用概率预测
            random_state=42
        )
        self._create_output_dir()

    def _create_output_dir(self):
        """创建输出目录"""
        os.makedirs(f'{self.out_dir}', exist_ok=True)

    def train(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        self.classes_ = self.model.classes_  # 记录类别信息

    def predict(self, X):
        """执行预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """获取预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """完整评估流程"""
        # 基础预测
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True, zero_division=0)
        }

        # 计算AUC（多类别处理）
        try:
            if len(self.classes_) == 2:
                metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics['auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr')
                metrics['auc_ovo'] = roc_auc_score(y, y_proba, multi_class='ovo')
        except Exception as e:
            print(f"AUC计算异常: {str(e)}")
            metrics['auc'] = None

        # 保存指标
        self._save_metrics(metrics)
        
        # 生成可视化
        self._plot_roc(y, y_proba)
        self._plot_pr(y, y_proba)
        self._plot_confusion_matrix(metrics['confusion_matrix'])
        
        return metrics

    def _save_metrics(self, metrics):
        """保存评估指标到JSON文件"""
        with open(f'{self.out_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    def _plot_roc(self, y_true, y_proba):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        # 多类别处理
        for i, class_id in enumerate(self.classes_):
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, i], pos_label=class_id)
            auc = roc_auc_score(
                (y_true == class_id).astype(int), 
                y_proba[:, i]
            )
            plt.plot(fpr, tpr, label=f'Class {class_id} (AUC = {auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.out_dir}/roc_curves.png')
        plt.close()

    def _plot_pr(self, y_true, y_proba):
        """绘制PR曲线"""
        plt.figure(figsize=(10, 8))
        
        # 多类别处理
        for i, class_id in enumerate(self.classes_):
            precision, recall, _ = precision_recall_curve(
                (y_true == class_id).astype(int),
                y_proba[:, i]
            )
            ap = average_precision_score(
                (y_true == class_id).astype(int),
                y_proba[:, i]
            )
            plt.plot(recall, precision, label=f'Class {class_id} (AP = {ap:.2f})')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curves')
        plt.legend(loc="upper right")
        plt.savefig(f'{self.out_dir}/pr_curves.png')
        plt.close()

    def _plot_confusion_matrix(self, matrix):
        """绘制混淆矩阵热力图"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classes_,
            yticklabels=self.classes_
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(f'{self.out_dir}/confusion_matrix.png')
        plt.close()

# TEST
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # 加载数据
    data = load_iris()
    X, y = data.data, data.target
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    svm_model = SVM(kernel='rbf', C=1.0, gamma='scale')
    svm_model.train(X_train, y_train)
    
    # 评估模型
    metrics = svm_model.evaluate(X_test, y_test)
    print("评估指标已保存至output目录")
