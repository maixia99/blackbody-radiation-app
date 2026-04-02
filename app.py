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
    if T_k <= 0: return np.zeros_like(lam_um) # 绝对零度下无辐射
    lam_m = lam_um * 1e-6
    exponent = C2 / (lam_m * T_k)
    # 使用 np.where 防止极端高温/极短波长下的计算溢出
    return np.where(exponent < 700, (C1 * lam_m**-5) / (np.exp(exponent) - 1), 0)

def get_band_fraction(T_k, start, end):
    if T_k <= 0: return 0.0
    func = lambda l: planck_law(l, T_k)
    # 扩大积分上限到 1000 以适应各种极端温度
    total, _ = quad(func, 0.1, 1000, limit=100) 
    band, _ = quad(func, start, end, limit=100)
    return band / total if total > 0 else 0

# --- 侧边栏：输入控制 ---
st.sidebar.header("输入参数")
st.sidebar.info("本程序基于《传热学》黑体辐射理论，实现了普朗克、维恩、斯忒藩-玻尔兹曼等定律的关联可视化。")

# 【修改点 1】：使用 number_input 替代 slider，取消上限，保留绝对零度(-273.15)下限
temp_c = st.sidebar.number_input("设置黑体温度 (°C)", min_value=-273.15, value=25.0, step=1.0)

# 【保持】：波长每次精确加减 0.1 μm，取消上限限制以自由探索
target_lam = st.sidebar.number_input("关注波长 (μm)", min_value=0.1, value=10.0, step=0.1)

T_k = temp_c + 273.15

# --- 主界面布局 ---
st.title("🌡️ 黑体辐射五大定律关联可视化")
st.markdown("基于《传热学》黑体辐射理论，实时分析温度与能量分布的关系。")

# --- 第一部分：核心指标 (Metrics) ---
col1, col2, col3 = st.columns(3)
if T_k > 0:
    lam_max = B_WIEN / T_k
    total_eb = SIGMA * T_k**4
    current_val = planck_law(target_lam, T_k)
else:
    lam_max, total_eb, current_val = 0, 0, 0

with col1:
    st.metric("维恩位移 (峰值波长)", f"{lam_max:.2f} μm")
    st.caption("依据：维恩位移定律 λm·T = b")
with col2:
    st.metric("总辐射力 (Eb)", f"{total_eb:.2f} W/m²")
    st.caption("依据：斯忒藩-玻尔兹曼定律 Eb = σT⁴")
with col3:
    st.metric(f"{target_lam:.1f}μm 处辐射力", f"{current_val:.2e}")
    st.caption("依据：普朗克定律全谱分布")

# --- 第二部分：动态光谱图 ---
st.subheader("📊 辐射光谱可视化 (普朗克曲线)")

# 扩大图表的波长展示范围 (0.1um 到 1000um)，适应极高和极低温度
lams = np.logspace(-1, 3, 800)
eb_vals = planck_law(lams, T_k)

fig = go.Figure()
if T_k > 0:
    # 普朗克曲线
    fig.add_trace(go.Scatter(x=lams, y=eb_vals, name="普朗克分布", line=dict(color='firebrick', width=3)))
    # 峰值标注
    fig.add_trace(go.Scatter(x=[lam_max], y=[planck_law(lam_max, T_k)], 
                             mode='markers+text', name="峰值点",
                             text=[f"峰值:{lam_max:.2f}μm"], textposition="top right",
                             marker=dict(color='red', size=10)))
# 用户输入波长标注
fig.add_vline(x=target_lam, line_dash="dash", line_color="green", annotation_text=f"关注点 {target_lam:.1f}μm")

fig.update_xaxes(type="log", title="波长 λ (μm)")
fig.update_yaxes(type="log", title="光谱辐射力 Ebλ (W/m³)")
fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=40))
st.plotly_chart(fig, use_container_width=True)

# --- 第三部分：能量分布饼图 ---
st.subheader("🍰 能量分布比例")
bands = [
    ("紫外/可见光", 0.1, 0.76),
    ("近/中红外", 0.76, 8.0),
    ("远红外", 8.0, 25.0),
    ("极远红外", 25.0, 1000.0) # 扩大计算区间
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
fig_pie = go.Figure(data=[go.Pie(labels=df_pie["波段"], values=df_pie["占比"], hole=.3)])
fig_pie.update_layout(height=400)
st.plotly_chart(fig_pie)

# --- 资料参考 ---
with st.expander("查看物理定律公式说明"):
    st.latex(r"E_{b\lambda} = \frac{C_1 \lambda^{-5}}{e^{C_2/(\lambda T)} - 1} \quad (\text{Planck's})")
    st.latex(r"\lambda_{max} T = 2897.8 \, \mu m \cdot K \quad (\text{Wien's})")
    st.latex(r"E_b = \sigma T^4 \quad (\text{Stefan-Boltzmann})")
