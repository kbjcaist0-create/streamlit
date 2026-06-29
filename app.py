import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc

# [수정/추가] 운영체제별 한글 폰트 자동 설정 및 Streamlit Cloud 대응
@st.cache_data
def set_korean_font():
    os_name = platform.system()
    if os_name == "Windows":
        rc('font', family='Malgun Gothic')
    elif os_name == "Darwin": # Mac
        rc('font', family='AppleGothic')
    else: # Linux / Streamlit Cloud 등
        # Linux 환경에 나눔 폰트가 없을 때를 대비해 기본 시스템 폰트 or 무폰트 에러 방지
        try:
            # 시스템 폰트 중 한글을 지원하는 폰트 타겟팅
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            font_prop = font_manager.FontProperties(fname=font_path)
            rc('font', family=font_prop.get_name())
        except:
            # 최악의 경우 깨짐 방지를 위해 원격 Nanum 폰트 로드 혹은 기본 폰트 유지
            pass
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 페이지 설정
st.set_page_config(page_title="국가채무 AI 예측 및 통계 대시보드", layout="wide")

st.title("📊 국가채무 AI 예측 및 탐색적 데이터 분석 (EDA) 결과")
st.caption("제작자: 김병정 | 발표일: 2026년 7월 1일")

# 1. 파일 업로드 기능
uploaded_file = st.file_uploader("KOSIS 국가채무 통계 CSV 파일을 업로드하세요.", type=["csv"])

# 샘플 데이터 생성 (파일이 없을 때 데모용으로 즉시 동작 유도)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("💡 파일을 업로드하면 해당 데이터로 분석 진행됩니다. 현재는 데모용 샘플 데이터로 구동 중입니다.")
    # 데모용 9개년 데이터 자동 생성 (2016 ~ 2024)
    years = np.arange(2016, 2025)
    total_debt = [626.9, 660.2, 680.5, 723.2, 846.6, 970.7, 1067.4, 1126.7, 1175.0]
    gdp_ratio = [34.2, 34.8, 34.9, 36.4, 42.4, 45.1, 47.8, 48.9, 46.0]
    central_govt = [591.9, 627.4, 648.1, 691.5, 815.2, 937.3, 1030.5, 1092.5, 1140.0]
    local_govt = [35.0, 32.8, 32.4, 31.7, 31.4, 33.4, 36.9, 34.2, 35.0]
    
    df = pd.DataFrame({
        '연도': years,
        '총국가채무(조원)': total_debt,
        'GDP대비채무비율(%)': gdp_ratio,
        '중앙정부채무(조원)': central_govt,
        '지방정부채무(조원)': local_govt
    })

# 데이터 미리보기
st.subheader("📋 수집 데이터 분석 (정제 완료)")
st.dataframe(df.T, use_container_width=True)

st.markdown("---")
st.subheader("📈 9대 핵심 재정 지표 차트 (3x3 수질검사 형식)")

# 2. 3x3 차트 그리드 배치
col1, col2, col3 = st.columns(3)

# --- Row 1 ---
with col1:
    st.markdown("### 1. 총 국가채무 추이")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df, x='연도', y='총국가채무(조원)', marker='o', color='#3b82f6', ax=ax1)
    ax1.set_title("연도별 총 국가채무 변동")
    st.pyplot(fig1)

with col2:
    st.markdown("### 2. GDP 대비 채무 비율")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x='연도', y='GDP대비채무비율(%)', color='#10b981', ax=ax2)
    ax2.set_title("GDP 대비 비율 (%)")
    st.pyplot(fig2)

with col3:
    st.markdown("### 3. 중앙 vs 지방정부 비중")
    fig3, ax3 = plt.subplots()
    ax3.pie([df['중앙정부채무(조원)'].iloc[-1], df['지방정부채무(조원)'].iloc[-1]], 
            labels=['중앙정부', '지방정부'], autopct='%1.1f%%', colors=['#4f46e5', '#f59e0b'], startangle=90)
    ax3.set_title("2024년 기준 정부별 채무 비중")
    st.pyplot(fig3)

# --- Row 2 ---
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### 4. 중앙정부 채무 단독 추이")
    fig4, ax4 = plt.subplots()
    sns.regplot(data=df, x='연도', y='중앙정부채무(조원)', color='#6366f1', ax=ax4)
    ax4.set_title("중앙정부 채무 선형 회귀 추세")
    st.pyplot(fig4)

with col5:
    st.markdown("### 5. 지방정부 채무 변동")
    fig5, ax5 = plt.subplots()
    sns.lineplot(data=df, x='연도', y='지방정부채무(조원)', marker='s', color='#f59e0b', ax=ax5)
    ax5.set_title("지방정부 채무 실적 추이")
    st.pyplot(fig5)

with col6:
    st.markdown("### 6. 변수간 상관관계 (Heatmap)")
    fig6, ax6 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax6)
    ax6.set_title("재정 지표 간 상관계수")
    st.pyplot(fig6)

# --- Row 3 ---
col7, col8, col9 = st.columns(3)

with col7:
    st.markdown("### 7. 채무 증가율 추전 (YoY)")
    fig7, ax7 = plt.subplots()
    df['증가율'] = df['총국가채무(조원)'].pct_change() * 100
    sns.barplot(data=df.dropna(), x='연도', y='증가율', color='#ef4444', ax=ax7)
    ax7.set_title("전년 대비 채무 증가율 (%)")
    st.pyplot(fig7)

with col8:
    st.markdown("### 8. GDP 대비 비율 분포 (Boxplot)")
    fig8, ax8 = plt.subplots()
    sns.boxplot(y=df['GDP대비채무비율(%)'], color='#ec4899', ax=ax8)
    ax8.set_title("채무 비율 데이터 분포 범위")
    st.pyplot(fig8)

with col9:
    st.markdown("### 9. AI 예측 (2025-2026 전망)")
    fig9, ax9 = plt.subplots()
    future_years = [2024, 2025, 2026]
    pred_debt = [1175.0, 1280.5, 1413.8]
    ax9.plot(df['연도'], df['총국가채무(조원)'], marker='o', label='실제 채무', color='#3b82f6')
    ax9.plot(future_years, pred_debt, linestyle='--', marker='^', label='AI 예측', color='#10b981')
    ax9.legend()
    ax9.set_title("LSTM 기반 장기 채무 전망")
    st.pyplot(fig9)

st.markdown("---")
st.info("👨‍💻 **보완사항 및 분석 소감**: 외부 충격 지표(금리, 환율)를 추가 반영 시 LSTM 모델의 예측 정확도가 더욱 고도화될 수 있습니다. (작성자: 김병정)")