import pandas as pd
import yaml
import base64
from pathlib import Path
from openai import OpenAI
from typing import Generator


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """获取图片MIME类型"""
    suffix = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return mime_types.get(suffix, "image/jpeg")

def load_data(filepath):
    """
    此函数用于加载数据集，支持 xlsx 和 csv 文件，会删除有缺失值的数据，默认最后一列是标签。

    参数:
    filepath (str): 数据集文件的路径，支持 xlsx 和 csv 格式。

    返回:
    tuple: 包含特征矩阵 X 和标签向量 y 的元组。
    """
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("不支持的文件格式，仅支持 .xlsx 和 .csv 文件。")
    
    # 删除包含缺失值的行
    df = df.dropna()
    
    # 提取特征和标签
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y

def load_openai_config():
    """加载LLM配置"""
    config_path = Path("../config.yaml") 
    if not config_path.exists():
        raise FileNotFoundError("找不到配置文件 config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config['llm_config']['OpenAI']['api_key'], config['llm_config']['OpenAI']['base_url']


def validate_messages(messages: list) -> list:
    """确保消息角色严格交替，合并连续的同角色消息"""
    if not messages:
        return []
    
    validated = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]
        content = msg["content"]
        
        # 合并连续的同角色消息
        j = i + 1
        while j < len(messages) and messages[j]["role"] == role:
            next_content = messages[j]["content"]
            # 合并内容
            if isinstance(content, list) and isinstance(next_content, list):
                content = content + next_content
            elif isinstance(content, list):
                content = content + [{"type": "text", "text": str(next_content)}]
            elif isinstance(next_content, list):
                content = [{"type": "text", "text": str(content)}] + next_content
            else:
                content = f"{content}\n{next_content}"
            j += 1
        
        validated.append({"role": role, "content": content})
        i = j
    
    return validated

def get_system_prompt(model_id: str) -> str:
    """根据智能体类型获取系统提示词"""
    prompts = {
        '疾病诊断': """你是一位专业的医疗AI助手，专注于疾病诊断辅助。你的职责包括：
- 根据用户描述的症状，分析可能的疾病类型
- 解读医学检查报告和影像资料
- 提供初步的诊断建议和就医指导
- 说明疾病的病因、症状特征和发展趋势

重要提醒：
1. 你提供的是辅助参考意见，不能替代专业医生的诊断
2. 遇到紧急或严重症状，请建议用户立即就医
3. 回答要专业、准确、通俗易懂
4. 保护用户隐私，不询问不必要的个人信息""",

        '健康管理': """你是一位专业的健康管理AI助手，专注于个人健康指导。你的职责包括：
- 分析用户的健康数据和生活习惯
- 提供个性化的健康改善建议
- 制定科学的运动计划和作息安排
- 解答慢性病管理和预防保健问题
- 提供心理健康和压力管理建议

重要提醒：
1. 建议要科学、实用、循序渐进
2. 考虑用户的实际情况，给出可执行的方案
3. 遇到需要医疗干预的情况，及时建议就医
4. 鼓励健康生活方式，但不要过度焦虑""",

        '营养指导': """你是一位专业的营养咨询AI助手，专注于膳食营养指导。你的职责包括：
- 分析用户的饮食结构和营养状况
- 提供个性化的膳食搭配建议
- 解答食物营养、饮食禁忌等问题
- 针对特定人群（孕妇、老人、儿童等）提供营养方案
- 帮助管理体重、改善亚健康状态

重要提醒：
1. 建议要科学合理，符合营养学原则
2. 考虑用户的口味偏好和实际条件
3. 特殊疾病患者的饮食建议需谨慎
4. 不推荐极端节食或不健康的减肥方法"""
    }
    return prompts.get(model_id, prompts['健康管理'])


def parse_response_content(content) -> str:
    """解析模型返回的内容，提取纯文本"""
    if isinstance(content, str):
        return content
    
    # 处理 [{'text': '...', 'type': 'text'}] 格式
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    texts.append(item['text'])
                elif 'content' in item:
                    texts.append(str(item['content']))
            else:
                texts.append(str(item))
        return ''.join(texts)
    
    return str(content) if content else ""


def stream_response(messages: list, model_id: str) -> Generator[str, None, None]:
    """流式响应生成器"""
    full_response = ""
    api_key, base_url = load_openai_config()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    mappings = {
       '疾病诊断': 'moonshot-v1-128k-vision-preview',
       '健康管理': 'moonshot-v1-128k-vision-preview',
       '营养指导': 'moonshot-v1-128k-vision-preview'
    }
    try:
        # 添加系统消息作为对话起点
        system_prompt = get_system_prompt(model_id)
        system_msg = [{"role": "system", "content": system_prompt}]
        validated_msgs = system_msg + validate_messages(messages)
        
        stream = client.chat.completions.create(
            model=mappings[model_id],
            messages=validated_msgs,
            stream=True,
            temperature=0.7,
        )
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                # 解析可能的多模态返回格式
                parsed = parse_response_content(delta_content)
                full_response += parsed
                yield full_response
    except Exception as e:
        yield f"API错误: {str(e)}"

def extract_image_path(content):
    """从 Gradio 内容格式中提取图片路径"""
    if isinstance(content, dict):
        # {"file": {"path": ...}} 格式
        if "file" in content and isinstance(content["file"], dict):
            return content["file"].get("path")
        # {"path": ...} 格式
        if "path" in content:
            return content["path"]
    return None


def build_api_content(content):
    """构建 API 调用的消息内容"""
    # 纯文本
    if isinstance(content, str):
        return content
    
    # 检查是否是图片格式
    image_path = extract_image_path(content)
    if image_path:
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        return [{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}]
    
    return str(content) if content else ""


def chat(history: list, model_id: str, image_cache: str = None) -> Generator[list, None, None]:
    """处理聊天交互"""
    messages = []
    
    # 转换历史记录格式
    for i, entry in enumerate(history):
        if isinstance(entry, dict):
            content = entry.get("content")
            role = entry.get("role")
            if content and role:
                # 确保 content 是字符串
                if isinstance(content, list):
                    # 如果是列表，提取文本
                    text_parts = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content]
                    content_str = " ".join(text_parts)
                else:
                    content_str = str(content)
                
                # 检查是否是最后一条用户消息且有图片缓存
                is_last_user_msg = (i == len(history) - 1 and role == "user" and image_cache)
                
                if is_last_user_msg:
                    # 构建多模态内容（图片+文本）
                    # 移除显示用的前缀
                    text = content_str.replace("📷 [已上传图片]\n", "")
                    base64_image = encode_image_to_base64(image_cache)
                    mime_type = get_image_mime_type(image_cache)
                    api_content = [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                        {"type": "text", "text": text}
                    ]
                else:
                    api_content = content_str
                
                messages.append({"role": role, "content": api_content})
        else:
            # 兼容旧格式
            if entry[0]: messages.append({"role": "user", "content": str(entry[0])})
            if entry[1]: messages.append({"role": "assistant", "content": str(entry[1])})
    
    # 流式响应
    response_generator = stream_response(messages, model_id)
    
    try:
        for partial_response in response_generator:
            yield history + [{"role": "assistant", "content": partial_response}]
    except Exception as e:
        yield history + [{"role": "assistant", "content": f"错误: {str(e)}"}]