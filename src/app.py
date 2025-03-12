import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from utils import load_data, chat
from models.svm import SVM
from models.logistic_regression import LogisticRegression
from models.random_forest import RandomForest
from plot import Visualizer  # 导入可视化类


# 全局变量存储模型
global_model = None

# 模型训练函数（使用自定义SVM）
def train_model(
    file, 
    model_type,
    # 随机森林参数
    rf_n_estimators=100,
    rf_max_depth=None,
    rf_max_features="sqrt",
    # SVM参数
    svm_kernel="linear",
    svm_C=1.0,
    svm_gamma="scale",
    # 逻辑回归参数 
    lr_penalty="l2",
    lr_C=1.0,
    lr_solver="lbfgs"
):
    global global_model
    try:
        X, y = load_data(file.name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "Random Forest":
            model = RandomForest(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth if rf_max_depth > 0 else None,
                max_features=rf_max_features
            )
        elif model_type == "SVM":
            model = SVM(
                kernel=svm_kernel,
                C=svm_C,
                gamma=svm_gamma
            )
        elif model_type == "Logistic Regression":
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
        
        global_model = model
        
        return f"训练完成！\n准确率: {acc:.3f}\n召回率: {rec:.3f}\nF1分数: {f1:.3f}"
    
    except Exception as e:
        return f"训练出错: {str(e)}"

# 预测函数
def make_prediction(pred_file):
    if global_model is None:
        return "⚠️ 请先训练模型！"
    
    try:
        X_pred, _ = load_data(pred_file.name)
        predictions = global_model.predict(X_pred)
        
        return pd.DataFrame({
            '样本序号': range(1, len(predictions)+1),
            '预测结果': predictions
        })
    
    except Exception as e:
        return f"预测出错: {str(e)}"
    
def evaluate_model(file):
    if global_model is None:
        return "⚠️ 请先训练模型！"
    if not file:
        return "⚠️ 请先上传文件！"
    # try:

    X, y = load_data(file.name)
    df = pd.DataFrame(X)
    df['标签'] = y
    # 提取类别
    classes = df['标签'].unique()
    viz = Visualizer(classes)
    y_proba = global_model.predict_proba(X)
    roc_fig = viz.plot_roc(y, y_proba)
    pr_fig = viz.plot_pr(y, y_proba)
    metrics = global_model.evaluate(X, y)
    confusion_matrix_fig = viz.plot_confusion_matrix(metrics['confusion_matrix'])
    return df, roc_fig, pr_fig, confusion_matrix_fig

                # 动态显示参数区的回调函数
def toggle_params(model_type):
    return {
        rf_params: gr.Accordion(visible=model_type == "Random Forest"),
        svm_params: gr.Accordion(visible=model_type == "SVM"),
        lr_params: gr.Accordion(visible=model_type == "Logistic Regression")
    }     
        
# 界面布局
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 1200px !important}") as demo:
    gr.Markdown("# 智能医疗系统")
    
    with gr.Row():
        # 左侧面板
        with gr.Column(scale=2):
            with gr.Tab("数据处理"):
                # TODO
                data_file = gr.File(
                    label="上传数据文件（CSV/XLSX）",
                    file_types=[".csv", ".xlsx"]
                )
                data_output = gr.DataFrame(
                    label="数据预览",
                    interactive=False,
                )
                preprocess_btn = gr.Button("数据预处理", variant="secondary")
                preprocess_output = gr.Textbox()
            with gr.Tab("模型训练"):
                train_file = gr.File(
                    label="上传训练文件（CSV/XLSX）",
                    file_types=[".csv", ".xlsx"]
                )
                model_choice = gr.Dropdown(
                    choices=["Random Forest", "SVM", "Logistic Regression"],
                    label="选择模型",
                    value="Random Forest"
                )
                # 各模型参数区
                with gr.Accordion("随机森林参数", visible=True) as rf_params:  # 默认显示
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
   
                model_choice.change(
                    fn=toggle_params,
                    inputs=model_choice,
                    outputs=[rf_params, svm_params, lr_params]
                )
                train_btn = gr.Button("开始训练", variant="primary") 
                train_output = gr.Textbox(
                    label="训练结果",
                    interactive=False,
                    placeholder="训练结果将显示在此处..."
                )
            with gr.Tab("模型评估"):
                # dataframe
                # 四个绘图区
                # 上传文件
                eval_file = gr.File(
                    label="上传评估文件（CSV/XLSX）",
                    file_types=[".csv", ".xlsx"]
                )
                # 开始评估按钮
                eval_btn = gr.Button("开始评估", variant="secondary")
                dataframe_component = gr.DataFrame(
                    label="模型指标",
                )
                roc_curve_plot = gr.Plot(label="ROC曲线")
                pr_curve_plot = gr.Plot(label="PR曲线")
                confusion_matrix_plot = gr.Plot(label="混淆矩阵")
                
                
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
                
            with gr.Tab("可视化"):
                pass
            #TODO
                

        # 右侧聊天面板
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="智能助手",
                height=400,
                bubble_full_width=False,
            )
            msg = gr.Textbox(
                label="输入消息",
                placeholder="输入问题后按回车发送",
                max_lines=3
            )
            model_id = gr.Dropdown(
                label="模型ID",
                value="医疗LLM",
                choices=["医疗LLM", "金融LLM", "教育LLM"]
            )
            clear_btn = gr.ClearButton([msg, chatbot], size='sm')
            def user(user_message, history):
                return "", history + [[user_message, None]]

    # 事件绑定
    train_btn.click(
        fn=train_model,
        inputs=[
            train_file, 
            model_choice,      # 模型类型
            # 随机森林参数
            rf_n_estimators, 
            rf_max_depth, 
            rf_max_features,
            # SVM参数
            svm_kernel,
            svm_C,
            svm_gamma,
            # 逻辑回归参数
            lr_penalty,
            lr_C,
            lr_solver
        ],
        outputs=train_output
    )

    
    eval_btn.click(
        evaluate_model,
        inputs=eval_file,
        outputs=[dataframe_component, roc_curve_plot, pr_curve_plot, confusion_matrix_plot]
    )
    
    pred_btn.click(
        make_prediction,
        inputs=pred_file,
        outputs=pred_output
    )

    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
    chat, [msg, chatbot, model_id], chatbot
)


if __name__ == "__main__":
    demo.launch()

