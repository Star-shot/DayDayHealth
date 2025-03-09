import numpy as np
import pandas as pd
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
    SVM classifier with evaluation metrics saving and visualization
    
    Params:
    kernel (str): kernel type, default is 'linear', options: 'linear', 'poly', 'rbf', 'sigmoid'
    C (float): regularization parameter, default is 1.0
    gamma (str/float): kernel coefficient, default is 'scale',options: 'scale', 'auto'
    out_dir (str): output directory, default is 'output'
    
    """
    def __init__(self, kernel='linear', C=1.0, gamma='scale', out_dir='output'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.out_dir = out_dir
        self.metrics = None
        self.model = svm.SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=True,  # activate probability estimates
                random_state=42
            )
        self._create_output_dir()

    def _create_output_dir(self):
        # make output directory if not exists
        os.makedirs(f'{self.out_dir}', exist_ok=True)

    def train(self, X, y):
        # train model
        self.model.fit(X, y)
        self.classes_ = self.model.classes_  # save classes

    def predict(self, X):
        # make prediction
        return self.model.predict(X)

    def predict_proba(self, X):
        # get probability estimates
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True, zero_division=0)
        }

        # calculate AUC
        try:
            if len(self.classes_) == 2:
                metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics['auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr')
                metrics['auc_ovo'] = roc_auc_score(y, y_proba, multi_class='ovo')
        except Exception as e:
            print(f"Error calculating AUC: {str(e)}")
            metrics['auc'] = None

        # self._save_metrics(metrics)
        # metrics in df
        metrics_df = pd.DataFrame(metrics, index=[0])
        # visualization
        self.plot_roc(y, y_proba)
        self.plot_pr(y, y_proba)
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        
        return metrics_df

    def _save_metrics(self, metrics):
        # save metrics to json file
        with open(f'{self.out_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    def plot_roc(self, y_true, y_proba):
        # plot ROC curve
        plt.figure(figsize=(10, 8))
        
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
        # plt.savefig(f'{self.out_dir}/roc_curves.png')
        plt.close()

    def plot_pr(self, y_true, y_proba):
        # plot PR curve
        plt.figure(figsize=(10, 8))
        
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
        # plt.savefig(f'{self.out_dir}/pr_curves.png')
        plt.close()

    def plot_confusion_matrix(self, matrix):
        # plot confusion matrix
        
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
        # plt.savefig(f'{self.out_dir}/confusion_matrix.png')
        plt.close()

# TEST
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # load data
    data = load_iris()
    X, y = data.data, data.target
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # train & evaluate
    svm_model = SVM(kernel='rbf', C=1.0, gamma='scale')
    svm_model.train(X_train, y_train)

    metrics = svm_model.evaluate(X_test, y_test)
    print(f'Evaluation metrics saved to{svm_model.out_dir}/metrics.json')
