import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

class Visualizer:
    def __init__(self, classes):
        self.classes_ = classes
    
    def plot_roc(self, y_true, y_proba):
        fig = go.Figure()
        
        for i, class_id in enumerate(self.classes_):
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, i], pos_label=class_id)
            auc = roc_auc_score((y_true == class_id).astype(int), y_proba[:, i])
            
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'Class {class_id} (AUC = {auc:.2f})',
                    line=dict(width=2)
                )
            )
        
        # 添加对角线
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='grey'),
                showlegend=False
            )
        )
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis_range=[0, 1],
            yaxis_range=[0, 1.05],
            width=800,
            height=600,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            ),
            hovermode="x unified"
        )
        return fig

    def plot_pr(self, y_true, y_proba):
        fig = go.Figure()
        
        for i, class_id in enumerate(self.classes_):
            precision, recall, _ = precision_recall_curve(
                (y_true == class_id).astype(int),
                y_proba[:, i]
            )
            ap = average_precision_score(
                (y_true == class_id).astype(int),
                y_proba[:, i]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'Class {class_id} (AP = {ap:.2f})',
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title='Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis_range=[0, 1],
            yaxis_range=[0, 1.05],
            width=800,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode="x unified"
        )
        return fig

    def plot_confusion_matrix(self, matrix):
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=self.classes_,
                y=self.classes_,
                colorscale='Blues',
                texttemplate="%{z}",
                textfont={"size":12}
            )
        )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=800,
            height=600,
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=-45)
        )
        return fig
