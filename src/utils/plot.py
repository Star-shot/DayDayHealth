import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

class Visualizer:
    def __init__(self, classes):
        self.classes_ = np.array(classes)
        self.n_classes = len(classes)
    
    def plot_roc(self, y_true, y_proba):
        """绘制 ROC 曲线"""
        fig = go.Figure()
        
        try:
            y_true = np.array(y_true)
            y_proba = np.array(y_proba)
            
            # 处理二分类情况
            if self.n_classes == 2:
                # 二分类只绘制正类的曲线
                if y_proba.ndim == 1:
                    proba_pos = y_proba
                else:
                    proba_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                
                fpr, tpr, _ = roc_curve(y_true, proba_pos)
                auc = roc_auc_score(y_true, proba_pos)
                
                fig.add_trace(
                    go.Scatter(
                        x=fpr.tolist(),
                        y=tpr.tolist(),
                        mode='lines',
                        name=f'AUC = {auc:.3f}',
                        line=dict(width=2, color='#1f77b4'),
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)'
                    )
                )
            else:
                # 多分类：为每个类别绘制曲线
                colors = px.colors.qualitative.Set1
                for i, class_id in enumerate(self.classes_):
                    y_binary = (y_true == class_id).astype(int)
                    fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                    auc = roc_auc_score(y_binary, y_proba[:, i])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=fpr.tolist(),
                            y=tpr.tolist(),
                            mode='lines',
                            name=f'类别 {class_id} (AUC={auc:.3f})',
                            line=dict(width=2, color=colors[i % len(colors)])
                        )
                    )
        except Exception as e:
            print(f"ROC 曲线绘制错误: {e}")
            fig.add_annotation(
                text=f"绘制失败: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # 添加对角线（随机分类器基准线）
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='grey', width=1),
                name='随机基准',
                showlegend=True
            )
        )
        
        fig.update_layout(
            title=dict(text='ROC 曲线', font=dict(size=16)),
            xaxis_title='假阳性率 (FPR)',
            yaxis_title='真阳性率 (TPR)',
            xaxis=dict(range=[0, 1], dtick=0.2, gridcolor='lightgrey'),
            yaxis=dict(range=[0, 1.02], dtick=0.2, gridcolor='lightgrey'),
            legend=dict(
                yanchor="bottom",
                y=0.02,
                xanchor="right",
                x=0.98,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            hovermode="closest",
            plot_bgcolor='white',
            margin=dict(l=60, r=30, t=50, b=60),
            autosize=True
        )
        return fig

    def plot_pr(self, y_true, y_proba):
        """绘制 PR 曲线"""
        fig = go.Figure()
        
        try:
            y_true = np.array(y_true)
            y_proba = np.array(y_proba)
            
            # 处理二分类情况
            if self.n_classes == 2:
                if y_proba.ndim == 1:
                    proba_pos = y_proba
                else:
                    proba_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                
                precision, recall, _ = precision_recall_curve(y_true, proba_pos)
                ap = average_precision_score(y_true, proba_pos)
                
                fig.add_trace(
                    go.Scatter(
                        x=recall.tolist(),
                        y=precision.tolist(),
                        mode='lines',
                        name=f'AP = {ap:.3f}',
                        line=dict(width=2, color='#2ca02c'),
                        fill='tozeroy',
                        fillcolor='rgba(44, 160, 44, 0.2)'
                    )
                )
                
                # 添加基准线（正类比例）
                baseline = float(np.sum(y_true) / len(y_true))
                fig.add_hline(
                    y=baseline, 
                    line_dash="dash", 
                    line_color="grey",
                    annotation_text=f"基准 ({baseline:.2f})",
                    annotation_position="bottom right"
                )
            else:
                # 多分类
                colors = px.colors.qualitative.Set1
                for i, class_id in enumerate(self.classes_):
                    y_binary = (y_true == class_id).astype(int)
                    precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, i])
                    ap = average_precision_score(y_binary, y_proba[:, i])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=recall.tolist(),
                            y=precision.tolist(),
                            mode='lines',
                            name=f'类别 {class_id} (AP={ap:.3f})',
                            line=dict(width=2, color=colors[i % len(colors)])
                        )
                    )
        except Exception as e:
            print(f"PR 曲线绘制错误: {e}")
            fig.add_annotation(
                text=f"绘制失败: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig.update_layout(
            title=dict(text='PR 曲线 (精确率-召回率)', font=dict(size=16)),
            xaxis_title='召回率 (Recall)',
            yaxis_title='精确率 (Precision)',
            xaxis=dict(range=[0, 1], dtick=0.2, gridcolor='lightgrey'),
            yaxis=dict(range=[0, 1.02], dtick=0.2, gridcolor='lightgrey'),
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            hovermode="closest",
            plot_bgcolor='white',
            margin=dict(l=60, r=30, t=50, b=60),
            autosize=True
        )
        return fig

    def plot_confusion_matrix(self, matrix):
        """绘制混淆矩阵"""
        fig = go.Figure()
        
        # 确保是 numpy array
        matrix = np.array(matrix)
        
        # 计算百分比用于显示
        matrix_percent = matrix / matrix.sum() * 100
        
        # 创建文本标签（数量 + 百分比）
        text_labels = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix[i])):
                row.append(f"{matrix[i][j]}<br>({matrix_percent[i][j]:.1f}%)")
            text_labels.append(row)
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[f'预测: {c}' for c in self.classes_],
                y=[f'实际: {c}' for c in self.classes_],
                colorscale='Blues',
                text=text_labels,
                texttemplate="%{text}",
                textfont={"size": 14},
                hovertemplate='实际: %{y}<br>预测: %{x}<br>数量: %{z}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=dict(text='混淆矩阵', font=dict(size=16)),
            xaxis_title='预测标签',
            yaxis_title='实际标签',
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed'),  # 反转Y轴使对角线从左上到右下
            margin=dict(l=80, r=30, t=50, b=80),
            autosize=True
        )
        return fig
