"""
UI 布局模块
定义 Gradio 界面组件
"""
import gradio as gr


# 样例数据路径
EXAMPLE_FILES = {
    "糖尿病分类数据": "../example/Diabetes Classification.csv",
    "体检数据": "../example/medical_examination.csv",
}


def create_data_processing_tab():
    """创建数据处理标签页"""
    with gr.Tab("数据处理"):
        with gr.Row():
            example_selector = gr.Dropdown(
                label="选择样例数据",
                choices=["自定义上传"] + list(EXAMPLE_FILES.keys()),
                value="糖尿病分类数据",
                scale=1
            )
            data_file = gr.File(
                label="上传数据文件（CSV/XLSX）",
                file_types=[".csv", ".xlsx"],
                value="../example/Diabetes Classification.csv",
                scale=2
            )
            data_info = gr.Textbox(label="数据信息", lines=2, scale=1)
        
        data_output = gr.DataFrame(label="数据预览", interactive=False)
        
        with gr.Accordion("数据分析", open=False):
            with gr.Tab("缺失值分析"):
                missing_btn = gr.Button("分析缺失值", size="sm")
                missing_info = gr.DataFrame(label="缺失值统计")
                missing_plot = gr.Plot(label="缺失值矩阵图")
            
            with gr.Tab("异常值分析"):
                outlier_btn = gr.Button("分析异常值", size="sm")
                outlier_plot = gr.Plot(label="箱线图")
            
            with gr.Tab("数据分布"):
                dist_btn = gr.Button("分析分布", size="sm")
                dist_plot = gr.Plot(label="分布图")
            
            with gr.Tab("相关性分析"):
                corr_btn = gr.Button("分析相关性", size="sm")
                corr_plot = gr.Plot(label="相关性热力图")
                high_corr_df = gr.DataFrame(label="高相关特征对 (|r| > 0.8)")
        
        gr.Markdown("### 数据预处理")
        with gr.Row():
            fill_strategy = gr.Dropdown(
                choices=["auto", "median", "mean", "mode", "knn", "drop"],
                value="auto",
                label="缺失值处理策略"
            )
            outlier_method = gr.Dropdown(
                choices=["cap", "drop", "median"],
                value="cap",
                label="异常值处理方法"
            )
        
        preprocess_btn = gr.Button("执行预处理", variant="primary")
        preprocess_output = gr.Textbox(label="处理日志", lines=4)
        
        with gr.Accordion("📊 数据分析报告", open=True):
            with gr.Row():
                download_data_btn = gr.Button("📥 下载数据", variant="secondary", size="sm")
                download_report_btn = gr.Button("📄 下载报告", variant="secondary", size="sm")
                send_to_llm_btn = gr.Button("🤖 发送给AI分析", variant="primary", size="sm")
            
            with gr.Row():
                encode_strategy = gr.Dropdown(
                    choices=["auto", "label", "onehot"],
                    value="auto",
                    label="分类变量编码策略",
                    info="auto: 自动选择 | label: 标签编码 | onehot: 独热编码",
                    scale=2
                )
                send_to_train_btn = gr.Button("🚀 传递到模型训练", variant="primary", size="sm", scale=1)
            
            with gr.Row():
                processed_file = gr.File(label="处理后数据", visible=False)
                report_file = gr.File(label="分析报告", visible=False)
            
            # 四张分析图表
            with gr.Row():
                with gr.Column():
                    auto_missing_plot = gr.Plot(label="缺失值分析")
                with gr.Column():
                    auto_outlier_plot = gr.Plot(label="异常值分析")
            
            with gr.Row():
                with gr.Column():
                    auto_dist_plot = gr.Plot(label="数据分布")
                with gr.Column():
                    auto_corr_plot = gr.Plot(label="相关性分析")
            
            # 文字报告
            with gr.Accordion("📝 详细报告", open=False):
                report_markdown = gr.Markdown(label="分析报告", value="*执行预处理后自动生成报告*")
            
            llm_prompt_state = gr.State(value="")
    
    return {
        'example_selector': example_selector,
        'data_file': data_file,
        'data_info': data_info,
        'data_output': data_output,
        'missing_btn': missing_btn,
        'missing_info': missing_info,
        'missing_plot': missing_plot,
        'outlier_btn': outlier_btn,
        'outlier_plot': outlier_plot,
        'dist_btn': dist_btn,
        'dist_plot': dist_plot,
        'corr_btn': corr_btn,
        'corr_plot': corr_plot,
        'high_corr_df': high_corr_df,
        'fill_strategy': fill_strategy,
        'outlier_method': outlier_method,
        'preprocess_btn': preprocess_btn,
        'preprocess_output': preprocess_output,
        'download_data_btn': download_data_btn,
        'download_report_btn': download_report_btn,
        'send_to_llm_btn': send_to_llm_btn,
        'encode_strategy': encode_strategy,
        'send_to_train_btn': send_to_train_btn,
        'processed_file': processed_file,
        'report_file': report_file,
        'auto_missing_plot': auto_missing_plot,
        'auto_outlier_plot': auto_outlier_plot,
        'auto_dist_plot': auto_dist_plot,
        'auto_corr_plot': auto_corr_plot,
        'report_markdown': report_markdown,
        'llm_prompt_state': llm_prompt_state,
    }


def create_model_training_tab():
    """创建模型训练标签页"""
    with gr.Tab("模型训练"):
        train_file = gr.File(
            label="上传训练文件（CSV/XLSX）",
            file_types=[".csv", ".xlsx"]
        )
        
        # 数据切分设置
        gr.Markdown("### 📊 数据切分设置")
        with gr.Row():
            split_method = gr.Radio(
                choices=["简单切分", "K折交叉验证"],
                value="简单切分",
                label="切分方式",
                scale=2
            )
            test_size = gr.Slider(
                0.1, 0.4, value=0.2, step=0.05,
                label="测试集比例",
                info="仅简单切分时有效",
                scale=1
            )
            k_folds = gr.Slider(
                3, 10, value=5, step=1,
                label="K折数",
                info="仅交叉验证时有效",
                visible=False,
                scale=1
            )
            random_seed = gr.Number(
                value=42,
                label="随机种子",
                precision=0,
                scale=1
            )
        
        gr.Markdown("### 🤖 模型选择")
        model_choice = gr.Dropdown(
            choices=["Random Forest", "SVM", "Logistic Regression"],
            label="选择模型",
            value="Random Forest"
        )
        # 各模型参数区
        with gr.Accordion("随机森林参数", visible=True) as rf_params:
            rf_n_estimators = gr.Slider(50, 500, value=100, step=50, label="树的数量 (n_estimators)")
            rf_max_depth = gr.Slider(2, 50, value=None, step=1, label="最大深度 (max_depth)")
            rf_max_features = gr.Dropdown(
                ["sqrt", "log2", 0.5, 0.8], 
                value="sqrt", 
                label="最大特征数 (max_features)"
            )
        
        with gr.Accordion("SVM参数", visible=False) as svm_params:
            svm_kernel = gr.Dropdown(
                ["linear", "poly", "rbf", "sigmoid"],
                value="linear",
                label="核函数 (kernel)"
            )
            svm_C = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="正则化强度 (C)")
            svm_gamma = gr.Dropdown(
                ["scale", "auto"],
                value="scale",
                label="核系数 (gamma)"
            )
        
        with gr.Accordion("逻辑回归参数", visible=False) as lr_params:
            lr_penalty = gr.Dropdown(
                ["l2", "l1", "elasticnet", "none"],
                value="l2",
                label="正则化类型 (penalty)"
            )
            lr_C = gr.Slider(0.01, 10.0, value=1.0, step=0.1, label="正则化强度 (C)")
            lr_solver = gr.Dropdown(
                ["lbfgs", "sag", "saga", "newton-cg", "liblinear"],
                value="lbfgs",
                label="优化算法 (solver)"
            )

        train_btn = gr.Button("开始训练", variant="primary") 
        train_output = gr.Textbox(
            label="训练结果",
            interactive=False,
            lines=8,
            placeholder="训练结果将显示在此处..."
        )
    
    return {
        'train_file': train_file,
        'split_method': split_method,
        'test_size': test_size,
        'k_folds': k_folds,
        'random_seed': random_seed,
        'model_choice': model_choice,
        'rf_params': rf_params,
        'rf_n_estimators': rf_n_estimators,
        'rf_max_depth': rf_max_depth,
        'rf_max_features': rf_max_features,
        'svm_params': svm_params,
        'svm_kernel': svm_kernel,
        'svm_C': svm_C,
        'svm_gamma': svm_gamma,
        'lr_params': lr_params,
        'lr_penalty': lr_penalty,
        'lr_C': lr_C,
        'lr_solver': lr_solver,
        'train_btn': train_btn,
        'train_output': train_output,
    }


def create_model_eval_tab():
    """创建模型评估标签页"""
    with gr.Tab("模型评估"):
        eval_status = gr.Markdown(
            value="💡 *训练模型后将自动使用测试集评估，或上传自定义评估数据*"
        )
        eval_file = gr.File(
            label="上传评估文件（可选，留空则使用训练时的测试集）",
            file_types=[".csv", ".xlsx"]
        )
        eval_btn = gr.Button("手动评估", variant="secondary")
        
        # 评估指标表格
        eval_metrics = gr.Dataframe(
            label="评估指标",
            headers=["指标", "值"],
            interactive=False
        )
        
        # 可视化图表（纵向排列）
        roc_curve_plot = gr.Plot(label="ROC曲线")
        pr_curve_plot = gr.Plot(label="PR曲线")
        confusion_matrix_plot = gr.Plot(label="混淆矩阵")
    
    return {
        'eval_status': eval_status,
        'eval_file': eval_file,
        'eval_btn': eval_btn,
        'eval_metrics': eval_metrics,
        'roc_curve_plot': roc_curve_plot,
        'pr_curve_plot': pr_curve_plot,
        'confusion_matrix_plot': confusion_matrix_plot,
    }


def create_prediction_tab():
    """创建批量预测标签页"""
    with gr.Tab("批量预测"):
        pred_file = gr.File(
            label="上传预测文件（CSV/XLSX）",
            file_types=[".csv", ".xlsx"]
        )
        pred_btn = gr.Button("开始预测", variant="secondary")
        pred_output = gr.Dataframe(
            label="预测结果",
            headers=["样本序号", "预测结果"],
            interactive=False
        )
    
    return {
        'pred_file': pred_file,
        'pred_btn': pred_btn,
        'pred_output': pred_output,
    }


def create_chat_panel():
    """创建聊天面板"""
    with gr.Column(scale=1):
        chatbot = gr.Chatbot(
            label="智能助手",
            height=400,
        )
        with gr.Row():
            msg = gr.Textbox(
                label="输入消息",
                placeholder="输入问题后按回车发送",
                max_lines=3,
                scale=4
            )
            img_input = gr.Image(
                label="上传图片",
                type="filepath",
                scale=1
            )
        model_id = gr.Dropdown(
            label="医疗智能体",
            value="健康管理",
            choices=["疾病诊断", "健康管理", "营养指导"]
        )
        provider_info = gr.Markdown(value="**当前模型**: qwen / qwen-max")
        with gr.Row():
            send_btn = gr.Button("发送", variant="primary", size='sm')
            clear_btn = gr.ClearButton([msg, chatbot, img_input], size='sm')
            download_history_btn = gr.Button("📥 导出对话", variant="secondary", size='sm')
        chat_history_file = gr.File(label="对话记录", visible=False)
        
        # 用于存储图片路径
        image_cache = gr.State(None)
    
    return {
        'chatbot': chatbot,
        'msg': msg,
        'img_input': img_input,
        'model_id': model_id,
        'provider_info': provider_info,
        'send_btn': send_btn,
        'clear_btn': clear_btn,
        'download_history_btn': download_history_btn,
        'chat_history_file': chat_history_file,
        'image_cache': image_cache,
    }


def create_layout():
    """创建完整的 UI 布局"""
    with gr.Blocks() as demo:
        gr.Markdown("# Starshot🌟")
        
        with gr.Row():
            # 左侧面板
            with gr.Column(scale=2):
                data_components = create_data_processing_tab()
                train_components = create_model_training_tab()
                eval_components = create_model_eval_tab()
                pred_components = create_prediction_tab()
                
                with gr.Tab("可视化"):
                    pass  # TODO
            
            # 右侧聊天面板
            chat_components = create_chat_panel()
        
        # 合并所有组件
        components = {}
        components.update(data_components)
        components.update(train_components)
        components.update(eval_components)
        components.update(pred_components)
        components.update(chat_components)
        
        return demo, components

