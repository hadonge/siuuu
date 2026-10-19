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
avg_audi_per_cluster = df.groupby('cluster')['total_audi'].mean().sort_values(ascending=False)
cluster_map = {old_cluster: new_name for old_cluster, new_name in zip(avg_audi_per_cluster.index, ['㉮', '㉯', '㉰'])}

df['cluster_label'] = df['cluster'].map(cluster_map)

# 5. 2차원 산점도
st.subheader("2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox("2차원 가로축 속성", options=list(feature_options.keys()), index=0, key='2d_x')
with col2:
    y_axis_2d = st.selectbox("2차원 세로축 속성", options=list(feature_options.keys()), index=1, key='2d_y')

fig_2d = px.scatter(
    df,
    x=feature_options[x_axis_2d],
    y=feature_options[y_axis_2d],
    color='cluster_label',
    hover_name='movieNm',
    labels={feature_options[x_axis_2d]: x_axis_2d, feature_options[y_axis_2d]: y_axis_2d, 'cluster_label': '묶음'},
    category_orders={'cluster_label': ['㉮', '㉯', '㉰']}
)
st.plotly_chart(fig_2d, use_container_width=True)

# 6. 3차원 산점도
st.subheader("3차원 산점도")
if len(selected_features_labels) >= 3:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox("3차원 X축 속성", options=list(feature_options.keys()), index=0, key='3d_x')
    with col4:
        y_axis_3d = st.selectbox("3차원 Y축 속성", options=list(feature_options.keys()), index=1, key='3d_y')
    with col5:
        z_axis_3d = st.selectbox("3차원 Z축 속성", options=list(feature_options.keys()), index=2 if len(feature_options) > 2 else 0, key='3d_z')

    fig_3d = px.scatter_3d(
        df,
        x=feature_options[x_axis_3d],
        y=feature_options[y_axis_3d],
        z=feature_options[z_axis_3d],
        color='cluster_label',
        hover_name='movieNm',
        labels={
            feature_options[x_axis_3d]: x_axis_3d,
            feature_options[y_axis_3d]: y_axis_3d,
            feature_options[z_axis_3d]: z_axis_3d,
            'cluster_label': '묶음'
        },
        category_orders={'cluster_label': ['㉮', '㉯', '㉰']}
    )
    # 점 크기 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("3차원 산점도를 표시하려면 선택된 속성이 3개 이상이어야 합니다.")

# 7. 묶음별 편수 및 네 속성의 평균(원래 단위) 표
st.subheader("묶음별 요약 정보 (원래 단위 평균)")

summary_df = df.groupby('cluster_label').agg(
    편수=('movieNm', 'count'),
    스크린_수_평균=('first_scrn', 'mean'),
    누적_관객_평균=('total_audi', 'mean'),
    십위권_일수_평균=('days_in_top10', 'mean'),
    롱런_지수_평균=('long_run_index', 'mean')
).reindex(['㉮', '㉯', '㉰'])

summary_df.columns = ['편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']
st.dataframe(summary_df.style.format({
    '편수': '{:,.0f}',
    '스크린 수 평균': '{:,.1f}',
    '누적 관객 평균': '{:,.1f}',
    '10위권 일수 평균': '{:,.1f}',
    '롱런 지수 평균': '{:,.2f}'
}))

# 8. 묶음별 누적 관객 Top 5 영화
st.subheader("묶음별 누적 관객 Top 5 영화")
cols = st.columns(3)
for idx, group_name in enumerate(['㉮', '㉯', '㉰']):
    with cols[idx]:
        st.markdown(f"### 묶음 {group_name}")
        top5_movies = (
            df[df['cluster_label'] == group_name]
            .sort_values(by='total_audi', ascending=False)
            .head(5)['movieNm']
            .tolist()
        )
        for rank, title in enumerate(top5_movies, 1):
            st.write(f"{rank}. {title}")

# 누적 관객 평균이 큰 묶음 순서대로 ㉮, ㉯, ㉰ 매핑
avg_audi_per_cluster = df.groupby('cluster')['total_audi'].mean().sort_

