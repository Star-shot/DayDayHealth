import pandas as pd
import yaml
from pathlib import Path
from openai import OpenAI
from typing import Generator

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
    """确保消息角色严格交替"""
    validated = []
    prev_role = None
    for msg in messages:
        current_role = msg["role"]
        if current_role == prev_role:
            # 自动插入相反角色消息
            insert_role = "assistant" if current_role == "user" else "user"
            validated.append({"role": insert_role, "content": "请继续"})
        validated.append(msg)
        prev_role = current_role
    return validated

def stream_response(messages: list, model_id: str) -> Generator[str, None, None]:
    """流式响应生成器"""
    full_response = ""
    api_key, base_url = load_openai_config()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    mappings = {
       '医疗LLM': 'moonshot-v1-128k', # 暂时占位
       '金融LLM': 'moonshot-v1-32k',
       '教育LLM': 'moonshot-v1-8k'
        }
    try:
        # 添加系统消息作为对话起点
        system_msg = [{"role": "system", "content": "你是一个有帮助的助手, 负责解决医疗问题"}]
        validated_msgs = system_msg + validate_messages(messages)
        
        stream = client.chat.completions.create(
            model=mappings[model_id],  # 指定专用模型
            messages=validated_msgs,
            stream=True,
            temperature=0.7,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                yield full_response
    except Exception as e:
        yield f"API错误: {str(e)}"

def chat(user_input: str, history: list, model_id: str) -> Generator[list, None, None]:
    """处理聊天交互"""
    messages = []
    
    # 转换历史记录格式
    for entry in history:
        # 跳过空回复
        if entry[0]: messages.append({"role": "user", "content": entry[0]})
        if entry[1]: messages.append({"role": "assistant", "content": entry[1]})
    
    # 添加当前输入
    if user_input: messages.append({"role": "user", "content": user_input})
    
    # 流式响应
    history.append([user_input, ""])
    response_generator = stream_response(messages, model_id)
    
    try:
        for partial_response in response_generator:
            history[-1][1] = partial_response
            yield history
    except Exception as e:
        history[-1][1] = f"错误: {str(e)}"
        yield history