import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 페이지 설정 (앱 제목 및 아이콘, 브라우저 탭 설정)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 필수 속성 결측치 및 조건 처리
    # 네 속성 중 값이 없거나 first_week_audi가 0인 영화 제외
    df_clean = df.dropna(subset=['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']).copy()
    df_clean = df_clean[df_clean['first_week_audi'] > 0]
    
    # 속성 변환 생성
    # 1) 스크린 수 상용로그
    df_clean['log_first_scrn'] = np.log10(df_clean['first_scrn'])
    # 2) 누적 관객 상용로그
    df_clean['log_total_audi'] = np.log10(df_clean['total_audi'])
    # 3) 10위권 일수 (그대로 사용)
    # 4) 롱런 지수 (누적 관객 / 첫 주 관객, 20 초과는 20으로 클리핑)
    df_clean['long_run_index'] = df_clean['total_audi'] / df_clean['first_week_audi']
    df_clean['long_run_index'] = df_clean['long_run_index'].clip(upper=20)
    
    return df, df_clean

raw_df, df = load_data()

# 편수 표시
st.write(f"전체 편수: **{len(raw_df)}**편 | 묶은 편수: **{len(df)}**편")

# 3. 군집화에 사용할 속성 선택 UI
feature_options = {
    '스크린 수 (로그)': 'log_first_scrn',
    '누적 관객 (로그)': 'log_total_audi',
    '10위권 일수': 'days_in_top10',
    '롱런 지수': 'long_run_index'
}

selected_features_labels = st.multiselect(
    "묶는 데 사용할 속성을 고르세요 (2개 이상 선택):",
    options=list(feature_options.keys()),
    default=list(feature_options.keys())
)

if len(selected_features_labels) < 2:
    st.warning("속성을 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_options[label] for label in selected_features_labels]

# 4. 표준화 및 K-Means 군집화 (k=3, random_state 고정)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df['cluster'] = cluster_labels

# 누적 관객 평균이 큰 묶음 순서대로 ㉮, ㉯, ㉰ 매핑
avg_audi_per_cluster = df.groupby('cluster')['total_audi'].mean().sort_
