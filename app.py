import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import quad

# --- 页面配置 ---
st.set_page_config(page_title="黑体辐射五大定律交互APP", layout="wide")

# --- 物理常数 ---
C1 = 3.7419e-16  # W·m^2
C2 = 1.4388e-2   # m·K
SIGMA = 5.67037e-8 # W/(m^2·K^4)
B_WIEN = 2897.8    # um·K

# --- 核心计算函数 ---
def planck_law(lam_um, T_k):
    if T_k <= 0: return np.zeros_like(lam_um)
    lam_m = lam_um * 1e-6
    exponent = C2 / (lam_m * T_k)
    return np.where(exponent < 700, (C1 * lam_m**-5) / (np.exp(exponent) - 1), 0)

def get_band_fraction(T_k, start, end):
    if T_k <= 0 or start >= end: return 0.0
    func = lambda l: planck_law(l, T_k)
    total, _ = quad(func, 0.1, 1000, limit=100) 
    band, _ = quad(func, start, end, limit=100)
    return band / total if total > 0 else 0

# --- 侧边栏：输入控制 ---
st.sidebar.header("输入参数设置")
st.sidebar.info("本程序基于《传热学》黑体辐射理论，实现了普朗克、维恩、斯忒藩-玻尔兹曼等定律的关联可视化。")

st.sidebar.subheader("1. 温度与单点波长")
temp_c = st.sidebar.number_input("设置黑体温度 (°C)", min_value=-273.15, value=25.0, step=1.0)
target_lam = st.sidebar.number_input("关注点波长 (μm)", min_value=0.1, value=10.0, step=0.1)

# 【新增功能】：自定义波长区间
st.sidebar.subheader("2. 自定义波段能量分析")
band_start = st.sidebar.number_input("波段起始波长 (μm)", min_value=0.1, value=8.0, step=0.1)
band_end = st.sidebar.number_input("波段结束波长 (μm)", min_value=band_start, value=13.0, step=0.1)

T_k = temp_c + 273.15

# --- 主界面布局 ---
st.title("🌡️ 黑体辐射五大定律关联可视化")
st.markdown("基于《传热学》黑体辐射理论，实时分析温度与能量分布的关系。")

# --- 第一部分：核心指标 (Metrics) ---
col1, col2, col3, col4 = st.columns(4)
if T_k > 0:
    lam_max = B_WIEN / T_k
    total_eb = SIGMA * T_k**4
    current_val = planck_law(target_lam, T_k)
    custom_band_frac = get_band_fraction(T_k, band_start, band_end)
else:
    lam_max, total_eb, current_val, custom_band_frac = 0, 0, 0, 0

with col1:
    st.metric("维恩位移 (峰值波长)", f"{lam_max:.2f} μm")
with col2:
    st.metric("总辐射力 (Eb)", f"{total_eb:.2f} W/m²")
with col3:
    st.metric(f"{target_lam:.1f}μm 处辐射力", f"{current_val:.2e}")
with col4:
    # 【新增功能】：显示选中波段的能量占比
    st.metric(f"选定波段 {band_start}-{band_end}μm 能量占比", f"{custom_band_frac * 100:.2f} %")

st.caption("定律依据：维恩位移定律(col1) | 斯忒藩-玻尔兹曼定律(col2) | 普朗克定律全谱分布(col3) | 普朗克公式波段积分(col4)")

# --- 第二部分：动态光谱图 ---
st.subheader("📊 辐射光谱可视化 (普朗克曲线与波段积分)")

lams = np.logspace(-1, 3, 800)
eb_vals = planck_law(lams, T_k)

fig = go.Figure()
if T_k > 0:
    # 普朗克主曲线
    fig.add_trace(go.Scatter(x=lams, y=eb_vals, name="普朗克分布", line=dict(color='firebrick', width=3)))
    
    # 【新增功能】：高亮用户选中的波段面积 (阴影填充)
    # 生成波段内的细密点阵以确保填充边界平滑
    lams_band = np.linspace(band_start, band_end, 150)
    eb_vals_band = planck_law(lams_band, T_k)
    fig.add_trace(go.Scatter(
        x=lams_band, y=eb_vals_band,
        fill='tozeroy', mode='none', # 填充到Y轴底部(Y=0)
        fillcolor='rgba(0, 176, 246, 0.4)', # 半透明蓝色
        name=f"波段 {band_start}-{band_end}μm"
    ))

    # 峰值标注
    fig.add_trace(go.Scatter(x=[lam_max], y=[planck_law(lam_max, T_k)], 
                             mode='markers+text', name="峰值点",
                             text=[f"峰值:{lam_max:.2f}μm"], textposition="top right",
                             marker=dict(color='red', size=10)))

# 单点波长线标注
fig.add_vline(x=target_lam, line_dash="dash", line_color="green", annotation_text=f"点λ={target_lam:.1f}")

fig.update_xaxes(type="log", title="波长 λ (μm)")
fig.update_yaxes(type="log", title="光谱辐射力 Ebλ (W/m³)")
fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=40), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- 第三部分：固定关键能量分布饼图 ---
st.subheader("🍰 全谱关键波段能量分布概览")
bands = [
    ("紫外/可见光", 0.1, 0.76),
    ("近/中红外", 0.76, 8.0),
    ("大气窗口(远红外)", 8.0, 14.0),
    ("极远红外", 14.0, 1000.0)
]

band_data = []
if T_k > 0:
    for name, start, end in bands:
        frac = get_band_fraction(T_k, start, end)
        band_data.append({"波段": name, "占比": frac})
else:
    for name, _, _ in bands:
        band_data.append({"波段": name, "占比": 0})

df_pie = pd.DataFrame(band_data)
fig_pie = go.Figure(data=[go.Pie(labels=df_pie["波段"], values=df_pie["占比"], hole=.4)])
fig_pie.update_layout(height=400)
st.plotly_chart(fig_pie)
