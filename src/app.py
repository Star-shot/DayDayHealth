"""
智能医疗系统主入口
Starshot🌟
"""
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 导入模块
from utils import load_data, chat, load_config
from utils.data_process import DataProcessor
from utils.plot import Visualizer
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
    select_example_data,
    send_to_training,
    global_model,
)
from models.svm import SVM
from models.logistic_regression import LogisticRegression
from models.random_forest import RandomForest


from web_design import create_layout, setup_events


# ==================== 模型训练相关 ====================

# 全局变量
_global_model = None
_global_test_data = None  # 保存测试集 (X_test, y_test)


def get_file_columns(file):
    """获取文件的列名，用于更新特征/标签选择下拉框"""
    if file is None:
        return gr.Dropdown(choices=[], value=[]), gr.Dropdown(choices=[], value=None)
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name, nrows=0)  # 只读取列名
        else:
            df = pd.read_excel(file.name, nrows=0)
        
        columns = df.columns.tolist()
        
        # 特征列：默认选择除最后一列外的所有列
        # 标签列：默认选择最后一列
        return (
            gr.Dropdown(choices=columns, value=columns[:-1] if len(columns) > 1 else []),
            gr.Dropdown(choices=columns, value=columns[-1] if columns else None)
        )
    except Exception as e:
        print(f"读取列名失败: {e}")
        return gr.Dropdown(choices=[], value=[]), gr.Dropdown(choices=[], value=None)


def train_model(
    file,
    feature_cols,
    label_col,
    split_method,
    test_size,
    k_folds,
    random_seed,
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
    """训练模型（支持自定义特征列和标签列）"""
    global _global_model, _global_test_data
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import numpy as np
    
    if file is None:
        return "请先上传训练数据！"
    
    try:
        # 加载数据
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        df = df.dropna()
        
        # 确定标签列
        if label_col and label_col in df.columns:
            y = df[label_col].values
        else:
            y = df.iloc[:, -1].values
            label_col = df.columns[-1]
        
        # 确定特征列
        if feature_cols and len(feature_cols) > 0:
            # 过滤掉不存在的列和标签列
            valid_features = [c for c in feature_cols if c in df.columns and c != label_col]
            if valid_features:
                X = df[valid_features].values
                feature_info = f"已选择 {len(valid_features)} 个特征"
            else:
                # 使用除标签外的所有列
                X = df.drop(columns=[label_col]).values
                feature_info = f"使用全部 {X.shape[1]} 个特征（默认）"
        else:
            # 使用除标签外的所有列
            X = df.drop(columns=[label_col]).values
            feature_info = f"使用全部 {X.shape[1]} 个特征（默认）"
        
        random_seed = int(random_seed) if random_seed else 42
        
        # 创建模型
        if model_type == "Random Forest":
            model = RandomForest(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                max_features=rf_max_features,
                random_state=random_seed
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
        
        # 检查类别分布，判断是否可以分层采样
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        # K折交叉验证
        if split_method == "K折交叉验证":
            k = int(k_folds)
            
            # 检查是否可以进行分层K折
            if min_class_count >= k:
                cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
                stratify_info = "分层"
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=k, shuffle=True, random_state=random_seed)
                stratify_info = "普通"
            
            # 使用底层 sklearn 模型进行交叉验证
            sklearn_model = model.model  # 获取底层 sklearn 模型
            
            acc_scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring='f1_macro')
            recall_scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring='recall_macro')
            
            # 划分一部分数据用于评估可视化
            if min_class_count >= 2:
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X, y, test_size=0.2, random_state=random_seed, stratify=y
                )
            else:
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X, y, test_size=0.2, random_state=random_seed
                )
            
            # 用全部数据训练最终模型
            model.train(X, y)
            _global_model = model
            _global_test_data = (X_eval, y_eval)  # 保存评估数据
            
            result = f"🔄 {k}折{stratify_info}交叉验证完成！\n\n"
            if stratify_info == "普通":
                result += f"⚠️ 部分类别样本过少（最小类别仅{min_class_count}个），已使用普通K折\n\n"
            result += f"📊 数据: {len(X)} 样本 × {X.shape[1]} 特征\n"
            result += f"   标签列: {label_col} | {feature_info}\n"
            result += f"   类别分布: {dict(class_counts)}\n\n"
            result += f"📊 准确率: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}\n"
            result += f"   各折: {', '.join([f'{s:.3f}' for s in acc_scores])}\n\n"
            result += f"📊 召回率: {recall_scores.mean():.3f} ± {recall_scores.std():.3f}\n"
            result += f"   各折: {', '.join([f'{s:.3f}' for s in recall_scores])}\n\n"
            result += f"📊 F1分数: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}\n"
            result += f"   各折: {', '.join([f'{s:.3f}' for s in f1_scores])}\n\n"
            result += f"✅ 最终模型已用全部数据训练，评估数据已保存（{len(X_eval)}样本）"
            
            return result
        
        # 简单切分
        else:
            # 检查是否可以分层采样（每个类别至少需要2个样本）
            if min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_seed, stratify=y
                )
                stratify_info = "（分层采样）"
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_seed
                )
                stratify_info = "（普通随机，部分类别样本过少）"
            
            model.train(X_train, y_train)
            _global_model = model
            _global_test_data = (X_test, y_test)  # 保存测试集用于评估
            
            result = f"✅ 简单切分训练完成！{stratify_info}\n\n"
            result += f"📊 数据: {len(X)} 样本 × {X.shape[1]} 特征\n"
            result += f"   标签列: {label_col} | {feature_info}\n\n"
            result += f"📊 数据切分:\n"
            result += f"   训练集: {len(X_train)} 样本 ({1-test_size:.0%})\n"
            result += f"   测试集: {len(X_test)} 样本 ({test_size:.0%})\n\n"
            result += f"📊 类别分布: {dict(class_counts)}\n\n"
            result += f"🔄 正在自动进行模型评估..."
            
            return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
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


def evaluate_model(file=None):
    """
    模型评估
    - 如果提供 file，使用上传的文件评估
    - 如果 file 为空，使用训练时保存的测试集
    """
    global _global_model, _global_test_data
    
    if _global_model is None:
        empty_df = pd.DataFrame({"提示": ["请先训练模型"]})
        return "⚠️ 请先训练模型！", empty_df, None, None, None
    
    # 获取评估数据
    if file is not None:
        X, y = load_data(file.name)
        data_source = "上传数据"
    elif _global_test_data is not None:
        X, y = _global_test_data
        data_source = "训练测试集"
    else:
        empty_df = pd.DataFrame({"提示": ["没有可用的评估数据"]})
        return "⚠️ 没有可用的评估数据，请上传文件或先训练模型", empty_df, None, None, None
    
    try:
        # 计算预测和指标
        preds = _global_model.predict(X)
        acc = accuracy_score(y, preds)
        rec = recall_score(y, preds, average='macro')
        f1 = f1_score(y, preds, average='macro')
        
        # 创建指标表格
        metrics_df = pd.DataFrame({
            "指标": ["准确率 (Accuracy)", "召回率 (Recall)", "F1分数 (F1-Score)", "样本数"],
            "值": [f"{acc:.4f}", f"{rec:.4f}", f"{f1:.4f}", str(len(y))]
        })
        
        # 可视化
        classes = np.unique(y)
        viz = Visualizer(classes)
        
        y_proba = _global_model.predict_proba(X)
        roc_fig = viz.plot_roc(y, y_proba)
        pr_fig = viz.plot_pr(y, y_proba)
        
        eval_metrics = _global_model.evaluate(X, y)
        confusion_matrix_fig = viz.plot_confusion_matrix(eval_metrics['confusion_matrix'])
        
        status = f"✅ 使用 **{data_source}** 评估完成 | 样本数: {len(y)}"
        
        return status, metrics_df, roc_fig, pr_fig, confusion_matrix_fig
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        empty_df = pd.DataFrame({"错误": [str(e)]})
        return f"❌ 评估出错: {str(e)}", empty_df, None, None, None


def toggle_params(model_type):
    """切换模型参数显示"""
    return (
        gr.Accordion(visible=model_type == "Random Forest"),
        gr.Accordion(visible=model_type == "SVM"),
        gr.Accordion(visible=model_type == "Logistic Regression")
    )


def toggle_split_params(split_method):
    """切换数据切分参数显示"""
    is_simple = split_method == "简单切分"
    return (
        gr.Slider(visible=is_simple),   # test_size
        gr.Slider(visible=not is_simple)  # k_folds
    )


# ==================== 创建应用 ====================

# 创建 UI 布局
demo, components = create_layout()

# 准备事件处理函数
handlers = {
    'select_example_data': select_example_data,
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
    'send_to_training': send_to_training,
    'update_provider_info': update_provider_info,
    'user_input_handler': user_input_handler,
    'chat': chat,
    'train_model': train_model,
    'make_prediction': make_prediction,
    'evaluate_model': evaluate_model,
    'toggle_params': toggle_params,
    'toggle_split_params': toggle_split_params,
    'get_file_columns': get_file_columns,
}

# 在 Blocks 上下文中设置事件绑定
with demo:
    setup_events(components, handlers)


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import sys
    
    PORT = 7860
    
    # 检查启动模式
    use_share = "--share" in sys.argv
    
    # 如果使用 share，自动配置 frpc
    if use_share:
        setup_frpc()
    
    print("=" * 50)
    print("🏥 智能医疗系统启动中...")
    print("=" * 50)
    
    public_url = None
    
    print(f"📍 本地访问: http://localhost:{PORT}")
    print(f"📍 局域网: http://0.0.0.0:{PORT}")
    print("=" * 50)
    print("启动参数:")
    print("  --share  使用 Gradio 内置分享(需网络支持)")
    print("=" * 50)
    
    # 获取项目根目录
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = os.path.join(project_root, "example")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=use_share,
        show_error=True,
        allowed_paths=[example_dir],  # 允许访问 example 文件夹
    )
