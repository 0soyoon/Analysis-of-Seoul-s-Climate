import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


@st.cache_data
def load_data(path="seoul_temperature.csv"):
    df = pd.read_csv(path)
    df["날짜"] = df["날짜"].astype(str).str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["일교차"] = df["최고기온(℃)"] - df["최저기온(℃)"]
    return df


def main():
    st.set_page_config(page_title="서울 기온 분석", layout="wide")
    st.title("서울 기온 분석 대시보드")

    df = load_data()
    years = sorted(df["연도"].unique())
    min_year, max_year = int(years[0]), int(years[-1])

    # 사이드바 연도 범위
    st.sidebar.header("연도 범위 선택")
    start_year, end_year = st.sidebar.slider(
        "연도",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )

    df_sel = df[(df["연도"] >= start_year) & (df["연도"] <= end_year)].copy()

    # ── 1. 연도별 평균기온 & 5년 이동평균 ──
    yearly = (
        df_sel.groupby("연도")["평균기온(℃)"]
        .mean()
        .reset_index()
        .rename(columns={"평균기온(℃)": "평균기온"})
        .sort_values("연도")
    )
    yearly["이동평균_5년"] = yearly["평균기온"].rolling(window=5, min_periods=1).mean()

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=yearly["연도"],
            y=yearly["평균기온"],
            mode="lines+markers",
            name="연평균 기온",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=yearly["연도"],
            y=yearly["이동평균_5년"],
            mode="lines",
            name="5년 이동평균",
            line=dict(color="#d62728", width=3, dash="dash"),
        )
    )
    fig1.update_layout(
        title=f"{start_year}년 – {end_year}년 서울 연도별 평균기온",
        xaxis_title="연도",
        yaxis_title="기온 (℃)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        height=500,
        hovermode="x unified",
    )
    fig1.update_xaxes(tickvals=yearly["연도"], tickfont=dict(size=11))
    st.plotly_chart(fig1, use_container_width=True)

    # ── 2. 월별 히트맵 ──
    st.subheader("월별 평균기온 히트맵")
    month_mean = (
        df_sel.groupby(["연도", "월"])["평균기온(℃)"]
        .mean()
        .reset_index()
        .rename(columns={"평균기온(℃)": "평균기온"})
    )
    pivot = month_mean.pivot(index="연도", columns="월", values="평균기온")
    # 모든 월(1~12) 컬럼 확보
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]
    # 행 전체에 NaN이 많은 연도 제거 옵션이 필요하면 여기서 처리 가능

    fig_heatmap = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlBu_r",
            hovertemplate="연도: %{y}<br>월: %{x}<br>평균기온: %{z:.1f}℃<extra></extra>",
            colorbar=dict(title="기온 (℃)"),
        )
    )
    fig_heatmap.update_layout(
        title="월별 평균기온",
        xaxis_title="월",
        yaxis_title="연도",
        height=500,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── 3. 최고기온 상위 10일 & 최저기온 하위 10일 ──
    st.subheader("최고기온 상위 10일")
    top10_high = df_sel.nlargest(10, "최고기온(℃)")[
        ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    ].reset_index(drop=True)
    top10_high["날짜"] = top10_high["날짜"].dt.strftime("%Y-%m-%d")
    st.dataframe(top10_high, use_container_width=True, hide_index=True)

    st.subheader("최저기온 하위 10일")
    bottom10_low = df_sel.nsmallest(10, "최저기온(℃)")[
        ["날짜", "지점", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
    ].reset_index(drop=True)
    bottom10_low["날짜"] = bottom10_low["날짜"].dt.strftime("%Y-%m-%d")
    st.dataframe(bottom10_low, use_container_width=True, hide_index=True)

    # ── 4. 일교차 연도별 평균 그래프 ──
    st.subheader("연도별 평균 일교차")
    daily_range_yearly = (
        df_sel.groupby("연도")["일교차"]
        .mean()
        .reset_index()
        .rename(columns={"일교차": "평균일교차"})
        .sort_values("연도")
    )

    fig_range = go.Figure()
    fig_range.add_trace(
        go.Scatter(
            x=daily_range_yearly["연도"],
            y=daily_range_yearly["평균일교차"],
            mode="lines+markers",
            name="평균 일교차",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6),
        )
    )
    fig_range.update_layout(
        title=f"{start_year}년 – {end_year}년 서울 연도별 평균 일교차",
        xaxis_title="연도",
        yaxis_title="일교차 (℃)",
        height=500,
        hovermode="x unified",
    )
    fig_range.update_xaxes(tickvals=daily_range_yearly["연도"], tickfont=dict(size=11))
    st.plotly_chart(fig_range, use_container_width=True)

    # ── 통계 카드 ──
    st.subheader("선택 구간 통계")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("평균 기온", f"{df_sel['평균기온(℃)'].mean():.1f} ℃")
    with c2:
        st.metric("최고 기온", f"{df_sel['최고기온(℃)'].max():.1f} ℃")
    with c3:
        st.metric("최저 기온", f"{df_sel['최저기온(℃)'].min():.1f} ℃")

    # ── 연도별 상세 표 ──
    st.subheader("연도별 상세")
    st.dataframe(
        yearly[["연도", "평균기온", "이동평균_5년"]],
        column_config={
            "연도": st.column_config.NumberColumn("연도", format="%d"),
            "평균기온": st.column_config.NumberColumn("연평균 기온 (℃)", format="%.1f"),
            "이동평균_5년": st.column_config.NumberColumn("5년 이동평균 (℃)", format="%.1f"),
        },
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
