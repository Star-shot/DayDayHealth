import streamlit as st
import numpy as np
from PIL import Image
import time
import io

# 初始化会话状态 - 替代Gradio的State组件
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {
        'last_text': '',
        'last_image': None,
        'process_time': None
    }

# 页面配置（必须放在最前面）
st.set_page_config(
    page_title="多功能处理系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义处理函数
def process_data(text_input, slider_value, image_input, choice):
    """处理输入数据并返回结果"""
    # 文本处理
    processed_text = f"处理结果: {text_input.upper()} | 参数: {slider_value}"
    
    # 图像处理
    if image_input is not None:
        img = Image.open(image_input)
        grayscale_image = img.convert('L')
    else:
        grayscale_image = None
        
    # 处理时间记录
    process_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    return processed_text, grayscale_image, process_time

# 侧边栏设置
with st.sidebar:
    st.header("控制面板")
    
    # 输入组件
    text_input = st.text_input("文本输入", placeholder="输入文字...")
    slider_value = st.slider("参数调节", 0, 100, 50, help="滑动选择数值")
    image_input = st.file_uploader("上传图片", type=['jpg', 'png'])
    choice = st.selectbox("模式选择", ["选项1", "选项2", "选项3"])
    
    # 示例数据按钮
    if st.button("加载示例数据"):
        text_input = "示例文本"
        slider_value = 75
        # 此处需要实际示例图片路径
        # image_input = "example.jpg"
        st.experimental_rerun()

# 主内容区域
st.title("数据处理中心")
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("输入预览")
    with st.expander("原始数据", expanded=True):
        # 实时显示输入数据
        st.write(f"当前文本: `{text_input}`")
        st.write(f"滑块数值: {slider_value}")
        if image_input:
            st.image(image_input, caption="上传的图片", use_column_width=True)

with col2:
    st.subheader("处理结果")
    
    # 处理按钮
    if st.button("开始处理", type="primary"):
        # 执行处理函数
        result_text, result_image, process_time = process_data(
            text_input, slider_value, image_input, choice
        )
        
        # 更新会话状态
        st.session_state.processed_data.update({
            'last_text': result_text,
            'last_image': result_image,
            'process_time': process_time
        })
    
    # 显示处理结果
    if st.session_state.processed_data['last_text']:
        with st.container():
            st.success("处理完成!")
            
            # 文本结果
            st.code(st.session_state.processed_data['last_text'], language='text')
            
            # 图像结果
            if st.session_state.processed_data['last_image']:
                st.image(
                    st.session_state.processed_data['last_image'],
                    caption="灰度处理结果",
                    use_column_width=True
                )
            
            # 时间戳
            st.caption(f"处理时间: {st.session_state.processed_data['process_time']}")
            
            # 下载按钮
            img_bytes = io.BytesIO()
            st.session_state.processed_data['last_image'].save(img_bytes, format='PNG')
            st.download_button(
                label="下载处理图片",
                data=img_bytes.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )

# 高级功能
with st.expander("调试信息", expanded=False):
    st.write("会话状态内容:")
    st.json(st.session_state.processed_data)
    
    # 性能监控
    st.metric("当前内存占用", "1.2 GB", delta="-0.1 GB")
    
    # 缓存测试
    @st.cache_data
    def heavy_computation(input_val):
        time.sleep(2)  # 模拟耗时操作
        return input_val * 2
        
    if st.button("运行耗时计算"):
        result = heavy_computation(10)
        st.write(f"计算结果: {result}")

# 底部信息
st.markdown("---")
st.markdown("""
<style>
/* 自定义CSS样式 */
div[data-testid="stExpander"] div[role="button"] p {
    font-weight: 800;
    color: #2e86c1;
}
</style>
""", unsafe_allow_html=True)
