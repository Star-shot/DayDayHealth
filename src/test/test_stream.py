import gradio as gr
from openai import OpenAI
from typing import Generator

kimi_url = "https://api.moonshot.cn/v1"
kimi_key = "sk-Llg1MqZy0oucHMnNwQYgfutkzAXn6S4fKrL212GE4vs1FF81"
kimi_id = 'moonshot-v1-128k'

deepseek_url = "https://api.deepseek.com/v1"
deepseek_key = "sk-88614777ad0a452fb0f2e0e4dff29716"
deepseek_id = 'deepseek-reasoner'

# 新版API初始化方式
client = OpenAI(
    api_key=kimi_key,
    base_url=kimi_url
    )

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
    try:
        # 添加系统消息作为对话起点
        system_msg = [{"role": "system", "content": "你是一个有帮助的助手"}]
        validated_msgs = system_msg + validate_messages(messages)
        
        stream = client.chat.completions.create(
            model=model_id,  # 指定专用模型
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

# Gradio界面配置
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="输入消息")
    clear = gr.Button("清空历史")

    def user(user_message: str, history: list):
        return "", history + [[user_message, None]]

    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
        chat, [msg, chatbot], chatbot
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)