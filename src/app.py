import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from utils import load_data, chat, load_config
from utils.data_process import DataProcessor
from models.svm import SVM
from models.logistic_regression import LogisticRegression
from models.random_forest import RandomForest
from plot import Visualizer  # 导入可视化类


# 全局变量存储模型和数据处理器
global_model = None
global_processor = None


# ==================== 数据预处理函数 ====================

def load_preview_data(file):
    """加载数据并预览"""
    if file is None:
        return None, "请先上传文件"
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        info = f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列\n"
        info += f"缺失值总数: {df.isnull().sum().sum()}"
        return df.head(20), info
    except Exception as e:
        return None, f"加载失败: {str(e)}"


def analyze_data(file):
    """上传数据后自动分析（不执行预处理）"""
    global global_processor, global_report
    import os
    
    if file is None:
        return "", "", None, None, None, None
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        processor = DataProcessor(df)
        global_processor = processor
        
        # 确保输出目录存在
        os.makedirs("output/plots", exist_ok=True)
        
        # 生成四张分析图并保存
        plot_paths = {}
        
        try:
            missing_fig = processor.plot_missing_matrix()
            missing_fig.savefig("output/plots/missing.png", dpi=100, bbox_inches='tight')
            plot_paths['missing'] = "output/plots/missing.png"
        except:
            missing_fig = None
            plot_paths['missing'] = None
        
        try:
            outlier_fig = processor.plot_boxplot(normalize=True)
            outlier_fig.savefig("output/plots/outlier.png", dpi=100, bbox_inches='tight')
            plot_paths['outlier'] = "output/plots/outlier.png"
        except:
            outlier_fig = None
            plot_paths['outlier'] = None
        
        try:
            dist_fig = processor.plot_distribution()
            dist_fig.savefig("output/plots/distribution.png", dpi=100, bbox_inches='tight')
            plot_paths['distribution'] = "output/plots/distribution.png"
        except:
            dist_fig = None
            plot_paths['distribution'] = None
        
        try:
            corr_fig = processor.plot_correlation_heatmap()
            corr_fig.savefig("output/plots/correlation.png", dpi=100, bbox_inches='tight')
            plot_paths['correlation'] = "output/plots/correlation.png"
        except:
            corr_fig = None
            plot_paths['correlation'] = None
        
        # 生成报告（带图片路径）
        report = processor.generate_report()
        report['plot_paths'] = plot_paths
        global_report = report
        
        # 生成带图片的 Markdown 报告
        md_with_images = generate_markdown_with_images(report, plot_paths)
        report['markdown_with_images'] = md_with_images
        
        return (
            md_with_images,
            report['llm_prompt'],
            missing_fig,
            outlier_fig,
            dist_fig,
            corr_fig
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"分析失败: {str(e)}", "", None, None, None, None


def generate_markdown_with_images(report: dict, plot_paths: dict) -> str:
    """生成带图片的 Markdown 报告"""
    md = report['markdown']
    
    # 在报告末尾添加图片部分
    md += "\n\n## 📈 分析图表\n\n"
    
    if plot_paths.get('missing'):
        md += "### 缺失值分析\n"
        md += f"![缺失值分析](plots/missing.png)\n\n"
    
    if plot_paths.get('outlier'):
        md += "### 异常值分析\n"
        md += f"![异常值分析](plots/outlier.png)\n\n"
    
    if plot_paths.get('distribution'):
        md += "### 数据分布\n"
        md += f"![数据分布](plots/distribution.png)\n\n"
    
    if plot_paths.get('correlation'):
        md += "### 相关性分析\n"
        md += f"![相关性分析](plots/correlation.png)\n\n"
    
    return md


def get_missing_info(file):
    """获取缺失值信息"""
    if file is None:
        return None, None
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        processor = DataProcessor(df)
        missing_info = processor.get_missing_info()
        missing_fig = processor.plot_missing_matrix()
        
        return missing_info, missing_fig
    except Exception as e:
        return None, None


def get_outlier_info(file):
    """获取异常值信息并绘制箱线图"""
    if file is None:
        return None
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        processor = DataProcessor(df)
        fig = processor.plot_boxplot()
        return fig
    except Exception as e:
        return None


def get_distribution_info(file):
    """获取数据分布信息"""
    if file is None:
        return None
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        processor = DataProcessor(df)
        fig = processor.plot_distribution()
        return fig
    except Exception as e:
        return None


def get_correlation_info(file):
    """获取相关性热力图"""
    if file is None:
        return None, None
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        processor = DataProcessor(df)
        fig = processor.plot_correlation_heatmap()
        high_corr = processor.get_high_correlation_pairs()
        
        return fig, high_corr
    except Exception as e:
        return None, None


def process_data(file, fill_strategy, outlier_method):
    """执行数据预处理并重新生成分析图表"""
    global global_processor, global_report
    import os
    
    if file is None:
        return None, "请先上传文件", "", "", None, None, None, None
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        
        # 执行数据处理
        processor = DataProcessor(df)
        processor.fill_missing(strategy=fill_strategy)
        processor.handle_outliers(method=outlier_method)
        
        global_processor = processor
        
        # 确保输出目录存在
        os.makedirs("output/plots", exist_ok=True)
        
        # 用处理后的数据生成分析图表
        plot_paths = {}
        
        try:
            missing_fig = processor.plot_missing_matrix()
            missing_fig.savefig("output/plots/missing.png", dpi=100, bbox_inches='tight')
            plot_paths['missing'] = "output/plots/missing.png"
        except:
            missing_fig = None
            plot_paths['missing'] = None
        
        try:
            outlier_fig = processor.plot_boxplot(normalize=True)
            outlier_fig.savefig("output/plots/outlier.png", dpi=100, bbox_inches='tight')
            plot_paths['outlier'] = "output/plots/outlier.png"
        except:
            outlier_fig = None
            plot_paths['outlier'] = None
        
        try:
            dist_fig = processor.plot_distribution()
            dist_fig.savefig("output/plots/distribution.png", dpi=100, bbox_inches='tight')
            plot_paths['distribution'] = "output/plots/distribution.png"
        except:
            dist_fig = None
            plot_paths['distribution'] = None
        
        try:
            corr_fig = processor.plot_correlation_heatmap()
            corr_fig.savefig("output/plots/correlation.png", dpi=100, bbox_inches='tight')
            plot_paths['correlation'] = "output/plots/correlation.png"
        except:
            corr_fig = None
            plot_paths['correlation'] = None
        
        # 生成完整分析报告
        report = processor.generate_report()
        report['plot_paths'] = plot_paths
        
        # 生成带图片的 Markdown 报告
        md_with_images = generate_markdown_with_images(report, plot_paths)
        report['markdown_with_images'] = md_with_images
        global_report = report
        
        # 简短处理日志
        log = processor.get_processing_log()
        brief_report = f"✅ 数据预处理完成！\n\n处理步骤:\n"
        brief_report += "\n".join([f"• {item}" for item in log])
        brief_report += f"\n\n处理后数据: {processor.get_data().shape[0]} 行 × {processor.get_data().shape[1]} 列"
        
        return (
            processor.get_data().head(20), 
            brief_report, 
            md_with_images,
            report['llm_prompt'],
            missing_fig,
            outlier_fig,
            dist_fig,
            corr_fig
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"处理失败: {str(e)}", "", "", None, None, None, None


# 全局报告存储
global_report = None


def download_report():
    """下载分析报告（包含图片的 zip 包）"""
    global global_report
    import zipfile
    import shutil
    
    if global_report is None:
        return gr.File(visible=False)
    
    import os
    os.makedirs("output", exist_ok=True)
    
    # 写入带图片路径的 Markdown
    md_content = global_report.get('markdown_with_images', global_report['markdown'])
    md_path = "output/data_analysis_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 创建 zip 包（包含报告和图片）
    zip_path = "output/data_analysis_report.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加 Markdown 报告
        zipf.write(md_path, "data_analysis_report.md")
        
        # 添加图片
        plots_dir = "output/plots"
        if os.path.exists(plots_dir):
            for img_file in os.listdir(plots_dir):
                if img_file.endswith('.png'):
                    zipf.write(os.path.join(plots_dir, img_file), f"plots/{img_file}")
    
    return gr.File(value=zip_path, visible=True)


def prepare_for_llm(llm_prompt):
    """准备发送给 LLM 的内容，填入输入框供用户确认"""
    global global_report
    import os
    
    if not llm_prompt:
        return "⚠️ 请先上传数据或执行预处理以生成分析报告", None
    
    # 准备图片路径（合并成一张拼接图）
    image_path = None
    if global_report and global_report.get('plot_paths'):
        # 尝试拼接四张图为一张
        try:
            from PIL import Image
            
            plot_paths = global_report['plot_paths']
            images = []
            for key in ['missing', 'outlier', 'distribution', 'correlation']:
                path = plot_paths.get(key)
                if path and os.path.exists(path):
                    images.append(Image.open(path))
            
            if images:
                # 创建 2x2 拼接图
                widths = [img.width for img in images]
                heights = [img.height for img in images]
                max_w = max(widths) if widths else 400
                max_h = max(heights) if heights else 300
                
                # 创建画布
                combined = Image.new('RGB', (max_w * 2, max_h * 2), 'white')
                
                positions = [(0, 0), (max_w, 0), (0, max_h), (max_w, max_h)]
                for i, img in enumerate(images[:4]):
                    # 调整图片大小
                    img_resized = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
                    combined.paste(img_resized, positions[i])
                
                # 保存拼接图
                os.makedirs("output/plots", exist_ok=True)
                image_path = "output/plots/combined_analysis.png"
                combined.save(image_path)
        except Exception as e:
            print(f"图片拼接失败: {e}")
            # 如果拼接失败，使用第一张图
            for key in ['correlation', 'distribution', 'outlier', 'missing']:
                path = global_report.get('plot_paths', {}).get(key)
                if path and os.path.exists(path):
                    image_path = path
                    break
    
    # 返回文本和图片路径，填入输入框
    text = f"📊 数据分析请求:\n\n{llm_prompt}"
    return text, image_path


def download_chat_history(history):
    """下载对话历史"""
    import os
    import json
    from datetime import datetime
    
    if not history:
        return gr.File(visible=False)
    
    os.makedirs("output", exist_ok=True)
    
    # 生成 Markdown 格式的对话记录
    md_content = "# 💬 AI 对话记录\n\n"
    md_content += f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"
    
    for msg in history:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'user':
            md_content += f"### 👤 用户\n\n{content}\n\n"
        elif role == 'assistant':
            md_content += f"### 🤖 AI 助手\n\n{content}\n\n"
        
        md_content += "---\n\n"
    
    # 保存 Markdown 文件
    output_path = "output/chat_history.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 同时保存 JSON 格式（便于程序读取）
    json_path = "output/chat_history.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    return gr.File(value=output_path, visible=True)


def download_processed_data():
    """下载处理后的数据"""
    global global_processor
    
    if global_processor is None:
        return gr.File(visible=False)
    
    import os
    # 确保目录存在
    os.makedirs("output", exist_ok=True)
    
    # 保存到文件
    output_path = "output/processed_data.csv"
    global_processor.get_data().to_csv(output_path, index=False)
    return gr.File(value=output_path, visible=True)

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
with gr.Blocks() as demo:
    gr.Markdown("# Starshot🌟")
    
    with gr.Row():
        # 左侧面板
        with gr.Column(scale=2):
            with gr.Tab("数据处理"):
                with gr.Row():
                    data_file = gr.File(
                        label="上传数据文件（CSV/XLSX）",
                        file_types=[".csv", ".xlsx"]
                    )
                    data_info = gr.Textbox(label="数据信息", lines=2)
                
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
                # bubble_full_width=False,
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
            
            def update_provider_info(agent_id):
                """更新显示的提供商信息"""
                try:
                    config = load_config()
                    agent_config = config.get('agent_models', {}).get(agent_id, {})
                    provider = agent_config.get('provider', config.get('default_provider', 'kimi'))
                    model_type = agent_config.get('model', 'default')
                    
                    # 获取实际模型名称
                    providers = config.get('llm_providers', {})
                    model_name = providers.get(provider, {}).get('models', {}).get(model_type, 'unknown')
                    
                    return f"**当前模型**: {provider} / {model_name}"
                except:
                    return "**当前模型**: 配置加载失败"
            
            model_id.change(
                fn=update_provider_info,
                inputs=model_id,
                outputs=provider_info
            )
            
            # 用于存储图片路径，供 API 调用使用
            image_cache = gr.State(None)
            
            def user(user_message, image, history, img_cache):
                """处理用户输入（文本+图片）"""
                if not user_message and not image:
                    return "", None, history, img_cache
                
                new_history = list(history)
                
                # 构建显示文本
                if image:
                    text = user_message or "请分析这张图片"
                    display_text = f"📷 [已上传图片]\n{text}"
                    img_cache = image  # 缓存图片路径供 API 使用
                else:
                    display_text = user_message
                    img_cache = None
                
                # 添加用户消息（纯文本格式，Gradio 兼容）
                new_history.append({"role": "user", "content": display_text})
                    
                return "", None, new_history, img_cache

    # ==================== 事件绑定 ====================
    
    # 数据处理事件 - 上传后自动预览并分析
    data_file.change(
        fn=load_preview_data,
        inputs=data_file,
        outputs=[data_output, data_info]
    ).then(
        fn=analyze_data,
        inputs=data_file,
        outputs=[
            report_markdown,
            llm_prompt_state,
            auto_missing_plot,
            auto_outlier_plot,
            auto_dist_plot,
            auto_corr_plot
        ]
    )
    
    missing_btn.click(
        fn=get_missing_info,
        inputs=data_file,
        outputs=[missing_info, missing_plot]
    )
    
    outlier_btn.click(
        fn=get_outlier_info,
        inputs=data_file,
        outputs=outlier_plot
    )
    
    dist_btn.click(
        fn=get_distribution_info,
        inputs=data_file,
        outputs=dist_plot
    )
    
    corr_btn.click(
        fn=get_correlation_info,
        inputs=data_file,
        outputs=[corr_plot, high_corr_df]
    )
    
    preprocess_btn.click(
        fn=process_data,
        inputs=[data_file, fill_strategy, outlier_method],
        outputs=[
            data_output, 
            preprocess_output, 
            report_markdown, 
            llm_prompt_state,
            auto_missing_plot,
            auto_outlier_plot,
            auto_dist_plot,
            auto_corr_plot
        ]
    )
    
    download_data_btn.click(
        fn=download_processed_data,
        outputs=processed_file
    )
    
    download_report_btn.click(
        fn=download_report,
        outputs=report_file
    )
    
    # 发送给 LLM 分析：填入输入框和图片框，让用户确认后发送
    send_to_llm_btn.click(
        fn=prepare_for_llm,
        inputs=[llm_prompt_state],
        outputs=[msg, img_input]
    )
    
    # 模型训练事件
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

    # 文本回车发送
    msg.submit(user, [msg, img_input, chatbot, image_cache], [msg, img_input, chatbot, image_cache]).then(
        chat, [chatbot, model_id, image_cache], chatbot
    )
    # 按钮发送
    send_btn.click(user, [msg, img_input, chatbot, image_cache], [msg, img_input, chatbot, image_cache]).then(
        chat, [chatbot, model_id, image_cache], chatbot
    )
    
    # 导出对话历史
    download_history_btn.click(
        fn=download_chat_history,
        inputs=chatbot,
        outputs=chat_history_file
    )


def setup_frpc():
    """自动配置 frpc（从项目 bin 目录复制到 Gradio 缓存）"""
    import shutil
    from pathlib import Path
    
    # 项目 bin 目录中的 frpc
    project_root = Path(__file__).parent.parent
    src_frpc = project_root / "bin" / "frpc_linux_amd64_v0.3"
    
    # Gradio 缓存目录
    gradio_cache = Path.home() / ".cache" / "huggingface" / "gradio" / "frpc"
    dst_frpc = gradio_cache / "frpc_linux_amd64_v0.3"
    
    if src_frpc.exists():
        gradio_cache.mkdir(parents=True, exist_ok=True)
        if not dst_frpc.exists() or dst_frpc.stat().st_size != src_frpc.stat().st_size:
            shutil.copy2(src_frpc, dst_frpc)
            dst_frpc.chmod(0o755)
            print(f"✅ frpc 已配置: {dst_frpc}")
        return True
    return False


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
            # 如果有 authtoken，可以设置: ngrok.set_auth_token("YOUR_TOKEN")
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