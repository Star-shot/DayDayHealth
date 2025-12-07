"""
应用辅助函数模块
包含数据处理、报告生成、下载等功能
"""
import os
import gradio as gr
import pandas as pd
from utils.data_process import DataProcessor
from utils import load_config


# ==================== 全局变量 ====================
global_model = None
global_processor = None
global_report = None

# 样例数据路径
EXAMPLE_FILES = {
    "糖尿病分类数据": "../example/Diabetes Classification.csv",
    "体检数据": "../example/medical_examination.csv",
}


# ==================== 样例数据选择 ====================

def select_example_data(example_name):
    """选择样例数据"""
    if example_name == "自定义上传" or example_name not in EXAMPLE_FILES:
        return gr.File(value=None)
    
    file_path = EXAMPLE_FILES[example_name]
    # 转换为绝对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.normpath(os.path.join(base_dir, file_path))
    
    if os.path.exists(abs_path):
        return gr.File(value=abs_path)
    return gr.File(value=None)


# ==================== 数据加载与分析 ====================

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


# ==================== 单独分析函数 ====================

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


# ==================== 数据预处理 ====================

def process_data(file, fill_strategy, outlier_method):
    """执行数据预处理并重新生成分析图表"""
    global global_processor, global_report
    
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


# ==================== 下载功能 ====================

def download_report():
    """下载分析报告（包含图片的 zip 包）"""
    global global_report
    import zipfile
    
    if global_report is None:
        return gr.File(visible=False)
    
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


def download_chat_history(history):
    """下载对话历史"""
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
    
    os.makedirs("output", exist_ok=True)
    
    # 保存到文件
    output_path = "output/processed_data.csv"
    global_processor.get_data().to_csv(output_path, index=False)
    return gr.File(value=output_path, visible=True)


def send_to_training(encode_strategy: str = 'auto'):
    """
    将处理后的数据传递到模型训练模块
    
    Args:
        encode_strategy: 分类变量编码策略 (auto/label/onehot)
    """
    global global_processor
    
    if global_processor is None:
        return gr.File(value=None), "⚠️ 请先执行数据预处理"
    
    os.makedirs("output", exist_ok=True)
    
    # 复制一份处理器用于编码（不影响原数据）
    from utils.data_process import DataProcessor
    df_copy = global_processor.get_data().copy()
    encoder = DataProcessor(df_copy)
    
    # 检查是否有需要编码的分类列
    cat_cols = encoder.get_categorical_columns()
    encode_info = ""
    
    if cat_cols:
        # 执行分类变量编码
        encoder.encode_categorical(strategy=encode_strategy)
        encode_log = encoder.get_processing_log()
        encode_info = f"\n📊 编码信息: {encode_log[-1] if encode_log else '无'}"
    
    # 保存编码后的数据
    output_path = "output/processed_data_for_training.csv"
    encoder.get_data().to_csv(output_path, index=False)
    
    # 数据信息
    df = encoder.get_data()
    info = f"✅ 已加载预处理数据\n"
    info += f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列"
    
    if cat_cols:
        info += f"\n原分类列: {list(cat_cols.keys())}"
        info += encode_info
    else:
        info += "\n（无需编码的分类列）"
    
    return gr.File(value=output_path), info


# ==================== LLM 相关 ====================

def prepare_for_llm(llm_prompt):
    """准备发送给 LLM 的内容，填入输入框供用户确认"""
    global global_report
    
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


def user_input_handler(user_message, image, history, img_cache):
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


# ==================== 启动配置 ====================

def setup_frpc():
    """自动配置 frpc（从项目 bin 目录复制到 Gradio 缓存）"""
    import shutil
    from pathlib import Path
    
    # 项目 bin 目录中的 frpc
    project_root = Path(__file__).parent.parent.parent
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

