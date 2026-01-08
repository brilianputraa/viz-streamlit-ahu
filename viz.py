# ahu_app/viz.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import itertools  # ✅ 누락 임포트

# 공통 모듈에서 쓰는 맵/함수들 임포트
try:
    from .common import (
        절기_분류, 항목명_한글, 항목_열량맵핑, get_motor_device_kwh,
    )
except ImportError:
    from common import (
        절기_분류, 항목명_한글, 항목_열량맵핑, get_motor_device_kwh,
    )

__all__ = [
    "draw_season_year_line",
    "draw_overlay_by_shifted_datetime",
    "show_공조기별_총비용_요약",
    "show_항목별_소모비용",
    "BAND_RANGES_RAT",
    "BAND_RANGES_RAH",
    "add_band",
    "평균선추가",
]


def draw_season_year_line(
    df, y_col, title="",
    절기_리스트=None, 절기별_월맵=None, color_map=None,
    평균선_컬럼=None
):
    df = df.copy()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    df["절기"] = df["날짜"].apply(절기_분류)
    df["레전드"] = df["공조기"] + " | " + df["연도"].astype(str)

    if 절기_리스트 is None:
        절기_리스트 = ["혹한기", "간절기", "혹서기"]

    if 절기별_월맵 is None:
        절기별_월맵 = {
            "혹한기": [12, 1, 2, 3],
            "간절기": [4, 5, 10, 11],
            "혹서기": [6, 7, 8, 9],
        }

    if color_map is None:
        색상_리스트 = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
        고유_레전드 = df["레전드"].unique()
        color_map = {r: 색상_리스트[i % len(색상_리스트)] for i, r in enumerate(고유_레전드)}

    for 절기 in 절기_리스트:
        st.markdown(f"### {절기} [{', '.join(map(str, 절기별_월맵[절기]))}]")
        df_season = df[(df["절기"] == 절기) & (df["월"].isin(절기별_월맵[절기]))].copy()

        if df_season.empty:
            st.info(f"📭 {절기} 데이터 없음")
            continue

        fig = go.Figure()

        # ---- x축 구성
        if 절기 == "혹한기":
            # 12월은 2003, 그 외는 2004로 붙여서 연속 축 생성 (윤년 이슈 회피)
            df_season["정렬월일"] = df_season["날짜"].dt.strftime("%m-%d")
            df_season["정렬날짜"] = pd.to_datetime(
                df_season["정렬월일"].apply(lambda x: "2003-" + x if x.startswith("12") else "2004-" + x),
                errors="coerce"
            )
            df_season["레전드_정렬"] = df_season["공조기"] + " | " + df_season["연도"].astype(str)
            x축, x축타입 = "정렬날짜", "date"

        elif 절기 in ("간절기", "혹서기"):
            # 같은 연도 내 월-일 비교: 카테고리 축 사용
            df_season["정렬날짜"] = df_season["날짜"]                       # gap 계산용 실제 datetime
            df_season["정렬월일"] = df_season["정렬날짜"].dt.strftime("%m-%d")  # x 표시는 월-일
            df_season["레전드_정렬"] = df_season["레전드"]
            x축, x축타입 = "정렬월일", "category"

        # y 컬럼 준비 & 정렬
        # (그대로) 정렬/준비
        df_season["y_plot"] = pd.to_numeric(df_season[y_col], errors="coerce")
        df_season = df_season.dropna(subset=["y_plot"]).sort_values("정렬날짜")

        이미_그린_레전드 = set()

        for 레전드 in df_season["레전드_정렬"].unique():
            df_sub = df_season[df_season["레전드_정렬"] == 레전드].copy()
            df_sub = df_sub.sort_values("정렬날짜")

            # 🔧 간격 기반으로 '단절 임계값'을 동적으로 결정
            #   - 일 단위(≥24h)로 보이면: 1500분 (= 24h + 1h 여유)
            #   - 그 외: 기존 300분
            median_step = df_sub["정렬날짜"].diff().median()
            if pd.notna(median_step) and median_step >= pd.Timedelta(days=1):
                gap_threshold_min = 1500
            else:
                gap_threshold_min = 300

            # gap 그룹 계산
            df_sub["시간차분"] = df_sub["정렬날짜"].diff().dt.total_seconds().div(60)
            df_sub["gap_group"] = (df_sub["시간차분"] > gap_threshold_min).cumsum()

            # 그룹별로 선 그리기
            for _, g in df_sub.groupby("gap_group"):
                if g.empty:
                    continue
                hover_text = 레전드  # "AHUxx | YYYY"
                show_legend = hover_text not in 이미_그린_레전드
                fig.add_trace(go.Scatter(
                    x=g[x축],
                    y=g["y_plot"],
                    mode="lines",           # 선 + 점
                    line=dict(color=color_map.get(hover_text, None), width=2),
                    name=hover_text,
                    legendgroup=hover_text,
                    showlegend=show_legend,
                    connectgaps=True                # 내부 NaN은 연결
                ))
                이미_그린_레전드.add(hover_text)


        # ---- x축 설정(타입 보존)
        if x축타입 == "date":
            fig.update_xaxes(
                title="날짜",
                type="date",
                tickformat="%m-%d",
                showline=True, linecolor="black"
            )
        else:  # category
            정렬순서 = sorted(df_season["정렬월일"].unique())
            # tick 과밀 방지
            샘플링_간격 = max(1, len(정렬순서) // 15)
            fig.update_xaxes(
                title="날짜",
                type="category",
                categoryorder="array",
                categoryarray=정렬순서,
                tickvals=정렬순서[::샘플링_간격],
                tickangle=0,
                showline=True, linecolor="black"
            )

        # ---- 평균선(선택)
        if 평균선_컬럼 and 평균선_컬럼 in df_season and not df_season[평균선_컬럼].isnull().all():
            평균값 = df_season[평균선_컬럼].mean()
            fig.add_hline(
                y=평균값, line_dash="dot", line_color="red",
                annotation_text=f"절기 평균: {평균값:.1f}",
                annotation_position="top left"
            )

        # ---- Y축/레이아웃
        label = 항목명_한글.get(y_col.replace("_kWh", ""), y_col)
        label += " (kWh)" if y_col.endswith("_kWh") else " (만원)"
        fig.update_yaxes(showgrid=True, zeroline=False, showline=True, linewidth=1, linecolor="black")
        
        # ---- 경고 밴드 추가
        ahu = df_season["공조기"].iloc[0] if not df_season.empty else None
        if y_col in ["RAT", "환기온도"] and ahu in BAND_RANGES_RAT:
            for low, high in BAND_RANGES_RAT[ahu]:
                add_band(fig, low, high, label="경고구간")
        elif y_col in ["RAH", "환기습도"] and ahu in BAND_RANGES_RAH:
            for low, high in BAND_RANGES_RAH[ahu]:
                add_band(fig, low, high, label="경고구간")

        fig.update_layout(yaxis_title=label, title=title or "절기별 연도별 일일 총 비용")
        st.plotly_chart(fig, use_container_width=True, key=f"{title}_{절기}")


def draw_overlay_by_shifted_datetime(df, y_col, title="", color_map=None, 평균선_컬럼=None):

    df = df.copy()
    df["연도"] = df["datetime"].dt.year
    df["레전드"] = df["공조기"] + " | " + df["연도"].astype(str)

    if color_map is None:
        색상_리스트 = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
        색상_순환기 = itertools.cycle(색상_리스트)
        고유_레전드 = sorted(df["레전드"].unique())
        color_map = {레전드: next(색상_순환기) for 레전드 in 고유_레전드}

    for 연도 in sorted(df["연도"].unique()):
        df_year = df[df["연도"] == 연도]
        fig = go.Figure()

        for 레전드 in df_year["레전드"].unique():
            df_sub = df_year[df_year["레전드"] == 레전드].copy()
            df_sub = df_sub.sort_values("datetime")
            df_sub["시간차"] = df_sub["datetime"].diff().dt.total_seconds().div(60)
            df_sub["gap_group"] = (df_sub["시간차"] > 300).cumsum()

            show_legend = True  # ✅ 처음에만 레전드 표시
            for _, g in df_sub.groupby("gap_group"):
                if g.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=g["datetime"],
                    y=g[y_col],
                    mode="lines",
                    name=레전드,
                    line=dict(color=color_map.get(레전드, None)),
                    connectgaps=False,
                    showlegend=show_legend,
                    legendgroup=레전드
                ))
                show_legend = False  # ✅ 이후에는 숨김


        # ---- 평균선
        if 평균선_컬럼:
            평균 = df_year[y_col].mean()
            fig.add_hline(
                y=평균, line_dash="dot", line_color="red",
                annotation_text=f"{연도} 평균: {평균:.1f}",
                annotation_position="top left"
            )

        # ---- 경고 밴드 추가
        if "환기온도" in title or "RAT" in title:
            ahu = df_year["공조기"].iloc[0] if not df_year.empty else None
            if ahu in BAND_RANGES_RAT:
                for low, high in BAND_RANGES_RAT[ahu]:
                    add_band(fig, low, high, label="경고구간")

        elif "환기습도" in title or "RAH" in title:
            ahu = df_year["공조기"].iloc[0] if not df_year.empty else None
            if ahu in BAND_RANGES_RAH:
                for low, high in BAND_RANGES_RAH[ahu]:
                    add_band(fig, low, high, label="경고구간")


        y_label = "값"
        if "환기온도" in title or "RAT" in title:
            y_label = "온도(℃)"
        elif "환기습도" in title or "RAH" in title:
            y_label = "습도(%)"
        elif "비용" in y_col:
            y_label = "비용(만원)"
        elif "열량" in y_col or "kWh" in y_col:
            y_label = "열량(kWh)"
        elif "개도율" in y_col:
            y_label = "개도율(%)"

        fig.update_layout(
            title=f"{title} - {연도}년",
            xaxis_title="날짜", yaxis_title=y_label,
            xaxis=dict(showline=True, linecolor="black"),
            yaxis=dict(showline=True, linecolor="black")
        )
        fig.update_xaxes(
            title="날짜",
            type="date",
            showgrid=True,
            showline=True,
            linecolor="black"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_공조기별_총비용_요약(df_총비용: pd.DataFrame):
    import pandas as pd
    import numpy as np
    import re

    st.subheader("📋 공조기별 총비용 + 항목별 상세 (원)")

    # 1) 총비용(만원) -> 원
    df_base = df_총비용.copy()
    df_base = df_base.rename(columns={"총비용(만원)": "총비용(원)"})
    df_base["총비용(원)"] = (df_base["총비용(원)"] * 10000).round().astype("int64")

    # 2) 항목별 합계(만원) -> 원 (세부 항목 피벗)
    항목별 = st.session_state.get("비용총합", pd.DataFrame())
    wide = pd.DataFrame(index=df_base["공조기"])  # AHU 기준 인덱스 프레임
    if not 항목별.empty:
        piv = (항목별.pivot_table(index="공조기", columns="항목명",
                                  values="비용(만원)", aggfunc="sum")
                        .fillna(0.0))
        piv = (piv * 10000).round().astype("int64")  # 원 단위
        want_cols = ["냉수코일", "프리쿨러 냉수코일", "스팀코일", "제습 스팀코일", "전기"]
        piv = piv[[c for c in want_cols if c in piv.columns]]
        wide = piv.reindex(df_base["공조기"]).fillna(0).astype("int64")

    # 3) 병합/정렬
    df_show = df_base.set_index("공조기").join(wide, how="left").fillna(0)
    # ✅ 총비용을 항목별 합으로 재정의(불일치 제거)
    part_cols = [c for c in ["냉수코일","프리쿨러 냉수코일","스팀코일","제습 스팀코일","전기"] if c in df_show.columns]
    if part_cols:
        df_show["총비용(원)"] = df_show[part_cols].sum(axis=1).astype("int64")
    df_show = df_show.sort_values("총비용(원)", ascending=False)

    # 4) 'AHU03' -> '공조기-03' 형식으로 표시
    def _fmt_ahu(idx: str) -> str:
        m = re.search(r"AHU(\d+)", str(idx))
        return f"공조기-{m.group(1).zfill(2)}" if m else str(idx)
    df_show.index = df_show.index.map(_fmt_ahu)

    # 5) 전체 합계 행 추가(재계산 이후)
    sum_row = pd.DataFrame(df_show.sum(numeric_only=True)).T
    sum_row.index = ["전체 합계"]
    df_show = pd.concat([df_show, sum_row], axis=0)

    # 6) 공조기명을 컬럼으로 노출(인덱스는 숨김)
    df_show = df_show.reset_index().rename(columns={"index": "공조기"})
    df_show = df_show[["공조기"] + [c for c in df_show.columns if c != "공조기"]]

    # 7) 스타일: 컬럼 배경 + 0원은 빈칸
    컬럼색 = {
        "총비용(원)": "#e6ffe6",
        "프리쿨러 냉수코일": "#f0faff",
        "냉수코일": "#e6eeff",
        "제습 스팀코일": "#f5d9c6",
        "스팀코일": "#ffe6e6",
        "전기": "#fff5e6",
    }
    def _col_bg(col: pd.Series):
        color = 컬럼색.get(col.name, "")
        return [f"background-color: {color}"] * len(col)

    def money_fmt(v):
        if (pd.isna(v)) or (v == 0):
            return ""
        try:
            return f"{int(v):,} 원"
        except Exception:
            return ""

    money_cols = df_show.select_dtypes(include=[np.number]).columns.tolist()
    styled = (df_show.style
              .format({c: money_fmt for c in money_cols})
              .apply(_col_bg, subset=[c for c in df_show.columns if c in 컬럼색]))

    st.dataframe(styled, use_container_width=True, hide_index=True)

def show_항목별_소모비용(df_filtered, 선택공조기, 단가):
    항목비용리스트 = []
    # ✅ 항목별 최대열량 맵은 전역의 '항목_열량맵핑'을 그대로 사용
    for 항목 in ["CCV", "PC_CCV", "HCV", "DH_HCV"]:
        df = df_filtered[(df_filtered["항목명"] == 항목) & (df_filtered["공조기"].isin(선택공조기))].copy()
        if df.empty:
            continue

        # show_항목별_소모비용 내부 변경 예시
        df = df.sort_values(["공조기", "datetime"])
        df["시간간격"] = df.groupby("공조기")["datetime"].diff().dt.total_seconds().div(3600)
        df = df[df["시간간격"] > 0].copy()
        mx_map = 항목_열량맵핑[항목]
        v1 = df["값"].shift(1); v2 = df["값"]
        df["열량_kWh"] = ((v1 + v2)/2) * (df["공조기"].map(mx_map)) * df["시간간격"] / 100 / 860


        sum_df = df.groupby("공조기")["열량_kWh"].sum().reset_index()
        sum_df["열량_kcal"] = sum_df["열량_kWh"] * 860

        # ✅ 냉수/증기 단가 적용
        if 항목 in ["CCV", "PC_CCV"]:
            sum_df["ton"] = sum_df["열량_kcal"] / (2.3 * 4.187 * 1000)
            sum_df["비용(원)"] = sum_df["ton"] * 단가["냉수단가"]
        else:
            sum_df["ton"] = sum_df["열량_kcal"] / (495 * 0.4 * 1000)
            sum_df["비용(원)"] = sum_df["ton"] * 단가["증기단가"]

        sum_df["항목명"] = 항목
        항목비용리스트.append(sum_df[["공조기", "항목명", "비용(원)"]])

    # ✅ 전기(모터) 비용
    전기_리스트 = []
    for ahu in 선택공조기:
        motor_kwh, detail_kwh, detail_hours = get_motor_device_kwh(df_filtered, ahu)
        motor_cost = int(round(motor_kwh * 단가["전기단가"]))
        전기_리스트.append({"공조기": ahu, "항목명": "전기", "비용(원)": motor_cost})

    if 전기_리스트:
        항목비용리스트.append(pd.DataFrame(전기_리스트))

    if not 항목비용리스트:
        return pd.DataFrame()

    # ✅ 스택 막대 데이터 조합
    비용총합 = pd.concat(항목비용리스트, ignore_index=True)
    비용총합["비용(만원)"] = 비용총합["비용(원)"] / 10000
    비용총합["항목명"] = 비용총합["항목명"].map(항목명_한글).fillna(비용총합["항목명"])

    항목_색상맵 = {
        "프리쿨러 냉수코일": "#f0faff",
        "냉수코일": "#e6eeff",
        "제습 스팀코일": "#f5d9c6",
        "스팀코일": "#ffe6e6",
        "환기온도": "#d3ebac",
        "환기습도": "#b9cfca",
        "전기": "#fff5e6",
        "전력": "#fff5e6",
    }

    st.subheader("📊 공조기별 총 에너지 소모비용")
    fig = px.bar(
        비용총합,
        x="공조기",
        y="비용(만원)",
        color="항목명",
        color_discrete_map=항목_색상맵,
        barmode="stack",
        title="공조기별 총 에너지 소모비용 (만원)"
    )

    # 막대 위 총합 레이블
    총합_레이블 = 비용총합.groupby("공조기")["비용(만원)"].sum().reset_index()
    for _, row in 총합_레이블.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["공조기"]],
            y=[row["비용(만원)"]],
            mode="text",
            text=[f"{int(round(row['비용(만원)'])):,}만원"],
            textposition="top center",
            showlegend=False
        ))

    fig.update_traces(selector=dict(type="bar"), texttemplate='%{y:,.0f}만원', textposition='inside')
    fig.update_layout(yaxis_title="비용 (만원)", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["비용총합"] = 비용총합
    return 비용총합


BAND_RANGES_RAT = {
    "AHU01": [[18, 18.9], [24.1, 25]], "AHU02": [[18, 18.9], [24.1, 25]], "AHU03": [[18, 18.9], [24.1, 25]], "AHU04": [[18, 18.9], [24.1, 25]], "AHU05": [[18, 18.9], [24.1, 25]],
    "AHU06": [[18, 18.9], [24.1, 25]], "AHU07": [[18, 18.9], [24.1, 25]], "AHU08": [[18, 18.9], [24.1, 25]], "AHU09": [[18, 18.9], [24.1, 25]], "AHU10": [[18, 18.9], [24.1, 25]],
    "AHU11": [[18, 18.9], [24.1, 25]], "AHU12": [[18, 18.9], [24.1, 25]], "AHU13": [[18, 18.9], [23.1, 24]], "AHU14": [[18, 18.9], [24.1, 25]], 
    "AHU020": [[1, 1.9], [24.1, 25]], "AHU021": [[1, 1.9], [29.1, 30]], "AHU022": [[18, 18.9], [24.1, 25]], "AHU023": [[18, 18.9], [24.1, 25]],
    "AHU024": [[18, 18.9], [24.1, 25]], "AHU025": [[18, 18.9], [24.1, 25]], "AHU026": [[18, 18.9], [24.1, 25]],
    "AHU39": [[15, 15.9], [24.1, 25]], "AHU45": [[18, 22], [18, 22]]
}

BAND_RANGES_RAH = {
    "AHU01": [[70, 75]], "AHU02": [[70, 75]], "AHU03": [[70, 75]], "AHU05": [[70, 75]],
    "AHU06": [[70, 75]], "AHU07": [[70, 75]], "AHU09": [[70, 75]], "AHU10": [[70, 75]],
    "AHU11": [[70, 75]], "AHU13": [[60, 65]], "AHU14": [[70, 75]],
    "AHU020": [[65.1, 70]], "AHU022": [[70, 75]],
    "AHU024": [[70, 75]], "AHU025": [[70, 75]], "AHU026": [[70, 75]],"AHU45": [[18, 22], [18, 22]]
}


def add_band(fig, ymin, ymax, color="rgba(255,0,0,0.6)", label="밴드"):
    fig.add_shape(
        type="rect",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=ymin, y1=ymax,
        fillcolor=color,
        opacity=0.6,
        line_width=0,
        layer="below"
    )
    fig.add_annotation(
        xref="paper", x=0.01, y=(ymin+ymax)/2, yref="y",
        text=label, showarrow=False, font=dict(color="orange", size=12),
        bgcolor="rgba(255,255,255,0.7)", borderpad=2
    )
    return fig

def 평균선추가(fig, df, y컬럼):
        if y컬럼 in df.columns and not df[y컬럼].isnull().all():
            평균값 = df[y컬럼].mean()
            fig.add_hline(y=평균값, line_dash="dot", line_color="red",
                        annotation_text=f"평균: {평균값:.1f}", annotation_position="top left")
        return fig

