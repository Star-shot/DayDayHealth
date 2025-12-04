import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

# ==========================================
# 1. 全局配置与样式 (CSS)
# ==========================================
st.set_page_config(
    page_title="HealthGuard 智能健康评估系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 以实现“PDF 风格”的专业外观
st.markdown("""
<style>
    /* 全局字体与背景 */
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 卡片容器样式 */
    .css-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
    }
    
    /* 关键指标高亮 */
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
    }
    
    /* 风险标签 */
    .risk-tag-high { background-color: #ffcccc; color: #cc0000; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;}
    .risk-tag-med { background-color: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;}
    .risk-tag-low { background-color: #d4edda; color: #155724; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;}

    /* 分隔线 */
    hr { margin-top: 1rem; margin-bottom: 1rem; border: 0; border-top: 1px solid rgba(0,0,0,.1); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 模拟后端接口 (Mock Backend)
# ==========================================
class HealthBackend:
    """
    模拟后端数据处理类。
    在实际生产环境中，这里会调用数据库或 REST API。
    """
    def __init__(self):
        pass

    def get_user_profile(self, user_id):
        """获取用户基本信息"""
        # 模拟基于 PDF 的用户 "大帅哥"
        return {
            "id": user_id,
            "name": "大帅哥", # 来自 PDF
            "age": 19.9,
            "height_cm": 186.0,
            "gender": "男性",
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    def process_health_data(self, input_metrics):
        """
        处理输入的生理数据，计算衍生指标
        input_metrics: 包含体重、体脂率等的字典
        """
        # 单位转换：斤 -> 公斤 (为了计算通用性，展示时可转回)
        weight_kg = input_metrics['weight_jin'] / 2
        
        # 计算 BMI
        height_m = input_metrics['height_cm'] / 100
        bmi = weight_kg / (height_m ** 2)
        
        # 模拟计算总分 (基于各项指标)
        # 简单逻辑：BMI 越接近 22 越高分，体脂率越标准越高分
        score = 100 - abs(22 - bmi) * 2 - abs(15 - input_metrics['body_fat_percent'])
        score = min(max(int(score), 40), 99) # 限制在 40-99 之间

        # 构造完整的数据包
        data = {
            "metrics": {
                "weight_jin": input_metrics['weight_jin'],
                "weight_kg": weight_kg,
                "bmi": round(bmi, 1),
                "heart_rate": 102, # 来自 PDF 示例
                "body_fat_percent": input_metrics['body_fat_percent'],
                "muscle_mass_jin": input_metrics['muscle_mass_jin'],
                "bmr": 1830,
                "visceral_fat_level": 4,
            },
            "composition": {
                "water_jin": 99.0, # 水分
                "protein_jin": 26.0, # 蛋白质
                "fat_jin": input_metrics['weight_jin'] * (input_metrics['body_fat_percent']/100), # 脂肪量估算
                "minerals_jin": 9.4, # 无机盐
            },
            "segments": {
                # 模拟 PDF 中的节段分析 (左上, 右上, 躯干, 左下, 右下)
                "muscle_balance": [100, 102.8, 107.6, 109.3, 119.7],
                "fat_balance": [75, 62.5, 72.9, 40.0, 40.0]
            },
            "score": score,
            "body_age": 16, # PDF 数据
            "body_type": "低脂肪型" if input_metrics['body_fat_percent'] < 10 else "标准型"
        }
        return data

    def generate_ai_report(self, data):
        """
        模拟大模型生成分析报告
        """
        bmi = data['metrics']['bmi']
        fat = data['metrics']['body_fat_percent']
        score = data['score']
        
        # 简单的规则生成文案，实际会调用 GPT/Claude 接口
        risk_level = "低风险" if score > 80 else "中风险" if score > 60 else "高风险"
        
        report_content = f"""
### 🩺 智能健康分析报告

**综合评级**: <span class='risk-tag-{'low' if score > 80 else 'med' if score > 60 else 'high'}'>{risk_level}</span> (得分: {score})

#### 1. 核心风险预警
* **BMI 指数 ({bmi})**: {"处于标准区间。" if 18.5 <= bmi <= 24 else "偏离标准值，需注意体重管理。" }
* **体脂率 ({fat}%)**: {"属于非常优秀的运动员水平。" if fat < 10 else "处于正常范围。" if fat < 20 else "体脂略高，建议进行有氧运动。"}
* **肌肉量**: 骨骼肌含量较高，基础代谢优秀 ({data['metrics']['bmr']} kcal)。

#### 2. 原因深度分析
* **营养代谢**: 蛋白质含量充足 ({data['composition']['protein_jin']}斤)，说明日常饮食中优质蛋白摄入良好。
* **运动习惯**: 节段肌肉分析显示下肢肌肉发达，推测您有规律的腿部力量训练或跑步习惯。
* **水分平衡**: 身体水分含量为 {data['composition']['water_jin']}斤，处于标准区间，细胞代谢活跃。

#### 3. 专家建议 (AI Generated)
1.  **饮食建议**: 维持当前高蛋白饮食，但如果体脂过低，建议适当增加优质碳水（如糙米、燕麦）的摄入以维持激素水平。
2.  **运动处方**: 您的左下肢与右下肢肌肉量略有不平衡 (右腿更强)，建议增加单腿训练（如单腿硬拉、保加利亚深蹲）来纠正体态。
3.  **生活方式**: 心率偏高 (102 bpm)，建议监测静息心率，增加冥想或深呼吸练习以降低交感神经兴奋度。
        """
        return report_content

# 初始化后端
backend = HealthBackend()

# ==========================================
# 3. 前端页面布局 (Frontend Layout)
# ==========================================

# --- 侧边栏：模拟输入接口 ---
with st.sidebar:
    st.header("⚙️ 数据控制台")
    st.info("模拟后端接收到的实时体检数据")
    
    # 模拟用户输入变量 (这些通常来自传感器或数据库)
    input_height = st.number_input("身高 (cm)", value=186.0, step=0.5)
    input_weight = st.slider("体重 (斤)", 100.0, 250.0, 149.3, step=0.1) # 默认 PDF 数据
    input_bfp = st.slider("体脂率 (%)", 3.0, 40.0, 7.7, step=0.1)     # 默认 PDF 数据 7.7%
    input_muscle = st.slider("骨骼肌 (斤)", 50.0, 150.0, 77.2, step=0.1) # 默认 PDF 数据

    st.markdown("---")
    st.caption("Backend API Status: Online 🟢")
    
    # 构造输入数据包
    input_payload = {
        "height_cm": input_height,
        "weight_jin": input_weight,
        "body_fat_percent": input_bfp,
        "muscle_mass_jin": input_muscle
    }
    
    # 获取处理后的数据
    user_info = backend.get_user_profile("USER_001")
    health_data = backend.process_health_data(input_payload)

# --- 主页面 ---

st.title(f"📊 人体成分深度分析报告")
st.markdown(f"**用户ID:** {user_info['id']} | **姓名:** {user_info['name']} | **检测时间:** {user_info['test_time']}")

# 第一部分：概览仪表盘 (Top Section)
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    # 绘制得分仪表盘
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health_data['score'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "身体健康得分"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2ecc71" if health_data['score'] > 80 else "#f1c40f"},
            'steps': [
                {'range': [0, 60], 'color': "#f8f9fa"},
                {'range': [60, 85], 'color': "#e9ecef"},
                {'range': [85, 100], 'color': "#d4edda"}],
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown(f"<center><b>身体年龄:</b> {health_data['body_age']}岁 (实际: {user_info['age']:.1f})</center>", unsafe_allow_html=True)
    st.markdown(f"<center><b>体型判定:</b> {health_data['body_type']}</center>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # 关键指标卡片网格
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("关键生理指标")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric("体重 (Weight)", f"{health_data['metrics']['weight_jin']} 斤", delta=f"{health_data['metrics']['weight_kg']:.1f} kg", delta_color="off")
        st.metric("骨骼肌 (Muscle)", f"{health_data['metrics']['muscle_mass_jin']} 斤", "强壮")
        
    with m_col2:
        st.metric("BMI 指数", f"{health_data['metrics']['bmi']}", "标准")
        st.metric("基础代谢 (BMR)", f"{health_data['metrics']['bmr']} kcal", "高代谢")
        
    with m_col3:
        st.metric("体脂率 (PBF)", f"{health_data['metrics']['body_fat_percent']}%", "-超低", delta_color="inverse")
        st.metric("内脏脂肪等级", f"{health_data['metrics']['visceral_fat_level']}", "健康")
    st.markdown('</div>', unsafe_allow_html=True)

# 第二部分：图表分析 (Charts)
st.subheader("📈 多维度成分分析")

chart_c1, chart_c2 = st.columns(2)

with chart_c1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("**1. 身体成分构成 (重量分布)**")
    
    # 环形图数据
    comp_labels = ['水分', '蛋白质', '脂肪', '无机盐']
    comp_values = [
        health_data['composition']['water_jin'],
        health_data['composition']['protein_jin'],
        health_data['composition']['fat_jin'],
        health_data['composition']['minerals_jin']
    ]
    
    fig_pie = px.pie(values=comp_values, names=comp_labels, hole=0.4, 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with chart_c2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("**2. 节段分析 (肌肉 vs 脂肪均衡度)**")
    
    # 雷达图数据准备
    categories = ['左上肢', '右上肢', '躯干', '右下肢', '左下肢']
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=health_data['segments']['muscle_balance'],
        theta=categories,
        fill='toself',
        name='肌肉评估 (%)',
        line_color='#3498db'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=health_data['segments']['fat_balance'],
        theta=categories,
        fill='toself',
        name='脂肪评估 (%)',
        line_color='#e74c3c',
        opacity=0.5
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 140]
            )),
        showlegend=True,
        height=300,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 第三部分：AI 分析报告 (LLM Report)
st.subheader("🤖 AI 深度健康诊断报告")

# 使用容器包裹，模拟生成的文字流
report_container = st.container()

with report_container:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    # 获取分析内容
    ai_content = backend.generate_ai_report(health_data)
    
    # 展示内容
    st.markdown(ai_content, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("*免责声明：本报告由 AI 大模型基于您的数据生成，仅供参考，不作为医疗诊断依据。*")
    st.markdown('</div>', unsafe_allow_html=True)

# 第四部分：历史趋势 (模拟数据)
with st.expander("查看历史健康趋势 (History Trend)", expanded=False):
    # 模拟历史数据
    dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
    # 生成随机波动但总体平稳的数据
    history_df = pd.DataFrame({
        '日期': dates,
        '体重(斤)': np.random.uniform(145, 155, 6),
        '体脂率(%)': np.random.uniform(7, 12, 6)
    })
    
    fig_line = px.line(history_df, x='日期', y=['体重(斤)', '体脂率(%)'], markers=True,
                       title="近半年体质变化趋势")
    fig_line.update_layout(hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)