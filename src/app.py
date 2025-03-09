import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from utils import load_data, load_openai_config, chat
from models.svm import SVM  # 导入自定义SVM类
from openai import OpenAI


# 全局变量存储模型
global_model = None

# 模型训练函数（使用自定义SVM）
def train_model(file, model_type, kernel='linear', C=1.0, gamma='scale'):
    global global_model
    try:
        X, y = load_data(file.name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "Random Forest":
            model = RandomForestClassifier()
        elif model_type == "SVM":
            model = SVM(kernel=kernel, C=C, gamma=gamma)  # 使用自定义SVM
        else:
            model = LogisticRegression()
            
        model.train(X_train, y_train) if model_type == "SVM" else model.fit(X_train, y_train)
        
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
        
# 界面布局
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 1200px !important}") as demo:
    gr.Markdown("# 智能医疗系统")
    
    with gr.Row():
        # 左侧面板
        with gr.Column(scale=2):
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
                with gr.Accordion("SVM高级参数", open=False):
                    kernel = gr.Dropdown(
                        ["linear", "poly", "rbf", "sigmoid"],
                        value="linear",
                        label="核函数"
                    )
                    C = gr.Number(1.0, label="正则化参数 C")
                    gamma = gr.Dropdown(
                        ["scale", "auto"],
                        value="scale",
                        label="Gamma 参数"
                    )
                train_btn = gr.Button("开始训练", variant="primary") 
                train_output = gr.Textbox(
                    label="训练结果",
                    interactive=False,
                    placeholder="训练结果将显示在此处..."
                )
            
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
                value="moonshot-v1-128k",
                choices=["医疗LLM", "金融LLM", "教育LLM"]
            )
            clear_btn = gr.ClearButton([msg, chatbot], size='sm')
            def user(user_message, history):
                return "", history + [[user_message, None]]

    # 事件绑定
    train_btn.click(
        train_model,
        inputs=[train_file, model_choice, kernel, C, gamma],
        outputs=train_output 
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

