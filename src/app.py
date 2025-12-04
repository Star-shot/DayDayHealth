"""
智能医疗系统主入口
Starshot🌟
"""
import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 导入模块
from utils import load_data, chat, load_config
from utils.data_process import DataProcessor
from utils.app_helpers import (
    load_preview_data,
    analyze_data,
    get_missing_info,
    get_outlier_info,
    get_distribution_info,
    get_correlation_info,
    process_data,
    download_report,
    download_chat_history,
    download_processed_data,
    prepare_for_llm,
    update_provider_info,
    user_input_handler,
    setup_frpc,
    global_model,
)
from models.svm import SVM
from models.logistic_regression import LogisticRegression
from models.random_forest import RandomForest
from plot import Visualizer

from web_design import create_layout, setup_events


# ==================== 模型训练相关 ====================

# 全局模型变量（需要在本文件中使用）
_global_model = None


def train_model(
    file, 
    model_type,
    rf_n_estimators=100,
    rf_max_depth=None,
    rf_max_features="sqrt",
    svm_kernel="linear",
    svm_C=1.0,
    svm_gamma="scale",
    lr_penalty="l2",
    lr_C=1.0,
    lr_solver="lbfgs"
):
    """训练模型"""
    global _global_model
    
    if file is None:
        return "请先上传训练数据！"
    
    try:
        X, y = load_data(file.name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 根据模型类型创建模型
        if model_type == "Random Forest":
            model = RandomForest(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                max_features=rf_max_features,
                random_state=42
            )
        elif model_type == "SVM":
            model = SVM(
                kernel=svm_kernel,
                C=svm_C,
                gamma=svm_gamma
            )
        else:  # Logistic Regression
            model = LogisticRegression(
                penalty=lr_penalty,
                C=lr_C,
                solver=lr_solver
            )
        
        model.train(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')
        
        _global_model = model
        
        return f"训练完成！\n准确率: {acc:.3f}\n召回率: {rec:.3f}\nF1分数: {f1:.3f}"
    
    except Exception as e:
        return f"训练出错: {str(e)}"


def make_prediction(pred_file):
    """批量预测"""
    global _global_model
    
    if _global_model is None:
        return "⚠️ 请先训练模型！"
    
    try:
        X_pred, _ = load_data(pred_file.name)
        predictions = _global_model.predict(X_pred)
        
        return pd.DataFrame({
            '样本序号': range(1, len(predictions)+1),
            '预测结果': predictions
        })
    
    except Exception as e:
        return f"预测出错: {str(e)}"


def evaluate_model(file):
    """模型评估"""
    global _global_model
    
    if _global_model is None:
        return "⚠️ 请先训练模型！", None, None, None
    if not file:
        return "⚠️ 请先上传文件！", None, None, None

    X, y = load_data(file.name)
    df = pd.DataFrame(X)
    df['标签'] = y
    
    classes = df['标签'].unique()
    viz = Visualizer(classes)
    y_proba = _global_model.predict_proba(X)
    roc_fig = viz.plot_roc(y, y_proba)
    pr_fig = viz.plot_pr(y, y_proba)
    metrics = _global_model.evaluate(X, y)
    confusion_matrix_fig = viz.plot_confusion_matrix(metrics['confusion_matrix'])
    
    return df, roc_fig, pr_fig, confusion_matrix_fig


def toggle_params(model_type):
    """切换模型参数显示"""
    return {
        'rf_params': gr.Accordion(visible=model_type == "Random Forest"),
        'svm_params': gr.Accordion(visible=model_type == "SVM"),
        'lr_params': gr.Accordion(visible=model_type == "Logistic Regression")
    }


# ==================== 创建应用 ====================

# 创建 UI 布局
demo, components = create_layout()

# 准备事件处理函数
handlers = {
    'load_preview_data': load_preview_data,
    'analyze_data': analyze_data,
    'get_missing_info': get_missing_info,
    'get_outlier_info': get_outlier_info,
    'get_distribution_info': get_distribution_info,
    'get_correlation_info': get_correlation_info,
    'process_data': process_data,
    'download_report': download_report,
    'download_chat_history': download_chat_history,
    'download_processed_data': download_processed_data,
    'prepare_for_llm': prepare_for_llm,
    'update_provider_info': update_provider_info,
    'user_input_handler': user_input_handler,
    'chat': chat,
    'train_model': train_model,
    'make_prediction': make_prediction,
    'evaluate_model': evaluate_model,
    'toggle_params': toggle_params,
}

# 在 Blocks 上下文中设置事件绑定
with demo:
    setup_events(components, handlers)


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import sys
    
    PORT = 7860
    
    # 检查启动模式
    use_ngrok = "--ngrok" in sys.argv
    use_share = "--share" in sys.argv
    
    # 如果使用 share，自动配置 frpc
    if use_share:
        setup_frpc()
    
    print("=" * 50)
    print("🏥 智能医疗系统启动中...")
    print("=" * 50)
    
    public_url = None
    
    # 使用 ngrok 进行公网部署
    if use_ngrok:
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(PORT, "http")
            print(f"✅ Ngrok 公网链接: {public_url}")
        except Exception as e:
            print(f"❌ Ngrok 启动失败: {e}")
            print("提示: 可以在 https://ngrok.com 注册获取免费 token")
    
    print(f"📍 本地访问: http://localhost:{PORT}")
    print(f"📍 局域网: http://0.0.0.0:{PORT}")
    print("=" * 50)
    print("启动参数:")
    print("  --ngrok  使用 ngrok 创建公网链接")
    print("  --share  使用 Gradio 内置分享(需网络支持)")
    print("=" * 50)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=use_share,
        show_error=True,
    )
