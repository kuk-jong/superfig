# ============================================================
# [전남] 무화과 연간 경영 분석 시스템 (Streamlit 앱) - 통합본 v1.0
# - 입력: 지역, 온실 규격(단동/연동), 측벽 곡면(방풍벽 길이 d=한쪽 벌어짐),
#         생산량/단가, 투자비, 에너지 조건, (권장) 기상자료 CSV(date,tmin,tmax)
# - 출력: 겨울/여름/연간 매출·비용·순이익, 난방 가동시간 통계, 차트, CSV 다운로드
#
# ※ 목적: "상대비교 + 근거(기상자료)" 기반 경영 판단 지원(절대값 정밀모델 아님)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import date

# ------------------------------------------------------------
# 0) Streamlit 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="전남 무화과 경영 분석기", layout="wide")
st.title("🗺️ [전남] 무화과 연간 경영 분석 시스템 (시간별 시뮬레이션 + 기상근거)")
st.markdown(
    """
- **단순 ‘야간 14시간’ 가정이 아니라**, 하루 24시간 기온 변화를 계산해 **목표온도(예: 15℃) 미만 시간에만** 난방부하를 누적합니다.
- **기상자료 CSV(date,tmin,tmax)**를 올리면, 해당 자료를 그대로 근거로 사용합니다.
- 본 도구는 **상대비교/유형화** 목적이며, 절대 난방비의 정밀 예측 모델이 아닙니다.
"""
)
st.divider()

# ------------------------------------------------------------
# 1) 지역별 "간이" 기온 특성값(예비 모드용)
# - CSV 업로드가 없을 때만 사용(랜덤 없이 고정)
# ------------------------------------------------------------
REGION_DATA = {
    "영암군 (무화과 주산지)": {"base": 2.0, "amp": 8.0},
    "해남군": {"base": 2.2, "amp": 7.8},
    "목포시": {"base": 2.5, "amp": 7.5},
    "신안군": {"base": 3.0, "amp": 7.0},
    "진도군": {"base": 3.2, "amp": 6.8},
    "완도군": {"base": 3.5, "amp": 6.5},
    "무안군": {"base": 1.5, "amp": 8.2},
    "강진군": {"base": 2.0, "amp": 8.0},
    "장흥군": {"base": 1.8, "amp": 8.2},
    "여수시": {"base": 3.0, "amp": 7.0},
    "순천시": {"base": 1.5, "amp": 8.5},
    "광양시": {"base": 2.0, "amp": 8.0},
    "고흥군": {"base": 2.8, "amp": 7.2},
    "보성군": {"base": 1.0, "amp": 8.5},
    "나주시": {"base": 0.5, "amp": 9.0},
    "담양군": {"base": -0.5, "amp": 9.5},
    "곡성군": {"base": -1.0, "amp": 10.0},
    "구례군": {"base": -0.5, "amp": 9.8},
    "화순군": {"base": -1.0, "amp": 9.8},
    "장성군": {"base": -0.5, "amp": 9.5},
    "함평군": {"base": 1.0, "amp": 8.8},
    "영광군": {"base": 1.0, "amp": 8.8},
}

# ------------------------------------------------------------
# 2) 곡면 측벽(지면~측고) 원호 근사 함수
#    - 입력: Hs(측고), d(방풍벽 길이 = 한쪽 벌어짐)
#    - 출력: 곡선 길이 s
# ------------------------------------------------------------
def side_arc_length(Hs: float, d: float) -> float:
    """
    곡면 측벽(지면~측고)을 원호로 근사한 곡선 길이 s 계산(실무형).
    - Hs: 측고(m)
    - d : 방풍벽 길이(한쪽 벌어짐, m)
    """
    if d <= 0:
        return float(Hs)

    # 원호 근사(지면 접지점에서 곡선이 거의 수직으로 출발한다고 가정)
    R = (d**2 + Hs**2) / (2.0 * d)
    alpha = math.atan2(Hs, (d - R))
    delta = math.pi - alpha
    s = R * delta
    # 안전장치: 극단값 방지
    if not np.isfinite(s) or s <= 0:
        return float(Hs)
    return float(s)

# ------------------------------------------------------------
# 3) 단동/연동 외피면적 계산(연구·경영분석용 근사)
# ------------------------------------------------------------
def calc_surface_area(
    gh_type: str,
    span_count: int,
    gh_width: float,
    gh_length: float,
    gh_side_h: float,
    gh_ridge_h: float,
    wing_d: float,
    k_roof_multi: float
) -> dict:
    """
    단동/연동 외피면적 계산(상대비교 목적).
    - 연동: 외측 측벽 2면, 전/후면은 총폭 기준, 지붕은 k_roof_multi로 근사
    """
    span_count = max(1, int(span_count))
    roof_height = max(0.0, gh_ridge_h - gh_side_h)
    W_total = gh_width * span_count

    # 곡면 측벽 길이 s(지면~측고)
    s_side = side_arc_length(gh_side_h, wing_d)

    # (1) 측벽(외측 2면)
    area_side = 2.0 * gh_length * s_side

    # (2) 전/후면(박공): 총폭 기준(연동은 폭이 커짐)
    one_end_wall = (W_total * gh_side_h) + (0.5 * W_total * roof_height)
    area_end = 2.0 * one_end_wall

    # (3) 지붕면
    if gh_type.startswith("단동"):
        roof_slope_len = math.sqrt((gh_width / 2.0) ** 2 + roof_height ** 2)
        area_roof = 2.0 * roof_slope_len * gh_length
    else:
        # 연동 지붕은 형태가 복잡하므로 "총폭×길이×보정계수"로 근사
        area_roof = float(k_roof_multi) * gh_length * W_total

    surface_area = area_side + area_end + area_roof

    return {
        "surface_area": float(surface_area),
        "area_roof": float(area_roof),
        "area_side": float(area_side),
        "area_end": float(area_end),
        "W_total": float(W_total),
        "s_side": float(s_side),
        "roof_height": float(roof_height),
    }

# ------------------------------------------------------------
# 4) 기상 CSV 로딩(date,tmin,tmax)
# ------------------------------------------------------------
def load_weather_csv(uploaded_file):
    """
    CSV 컬럼 요구:
      - date: YYYY-MM-DD
      - tmin: 일최저(℃)
      - tmax: 일최고(℃)
    """
    df = pd.read_csv(uploaded_file)
    required = {"date", "tmin", "tmax"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV 컬럼이 부족합니다. 반드시 date,tmin,tmax 컬럼이 필요합니다.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
    df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
    df = df.dropna(subset=["tmin", "tmax"])
    return df

# ------------------------------------------------------------
# 5) 사이드바 입력 UI
# ------------------------------------------------------------
with st.sidebar:
    with st.form(key="input_form"):
        st.header("📝 데이터 입력")
        st.info("입력 후 맨 아래 **분석 실행** 버튼을 누르세요.")

        with st.expander("0. 지역 선택", expanded=True):
            region_name = st.selectbox("전남 시·군 선택", list(REGION_DATA.keys()))
            weather_file = st.file_uploader(
                "기상자료 CSV 업로드 (date,tmin,tmax) [권장]",
                type=["csv"]
            )
            st.caption("※ CSV가 없으면 ‘간이(가정) 기온모드’로 계산합니다(랜덤 없음).")

        with st.expander("1. 온실 규격", expanded=False):
            gh_type = st.radio("온실 형태", ["단동 (1동)", "연동 (여러 동 연결)"])
            span_count = st.number_input("연동 수", value=1 if gh_type == "단동 (1동)" else 3, step=1, min_value=1)
            gh_width = st.number_input("폭 (m)", value=6.0, step=0.5, min_value=1.0)
            gh_length = st.number_input("길이 (m)", value=50.0, step=1.0, min_value=1.0)
            gh_side_h = st.number_input("측고 Hs (m)", value=2.0, step=0.2, min_value=0.5)
            gh_ridge_h = st.number_input("동고 Hr (m)", value=3.5, step=0.2, min_value=0.5)

            # 측벽 곡면(지면~측고) 벌어짐: 방풍벽 길이(d) = 한쪽 벌어짐
            wing_d = st.number_input("방풍벽 길이 d (한쪽 벌어짐, m)", value=1.0, step=0.1, min_value=0.0)
            st.caption("※ d=0이면 측벽을 직선(수직)으로 간주합니다.")

            # 연동 지붕 보정계수(k_roof): 상대비교용
            if gh_type == "연동 (여러 동 연결)":
                k_roof_multi = st.selectbox("연동 지붕 보정계수(k_roof)", [1.08, 1.12, 1.18], index=1)
            else:
                k_roof_multi = 1.12  # 단동은 사용하지 않지만 변수 통일을 위해 둠

            # 바닥면적(참고)
            floor_area_m2 = gh_width * gh_length * int(span_count)
            floor_area_py = floor_area_m2 / 3.3
            st.caption(f"바닥면적(참고): {floor_area_m2:,.1f} ㎡ (≈ {floor_area_py:,.1f} 평)")

        with st.expander("2. 연간 생산 계획", expanded=False):
            st.markdown("**🌞 여름 작기**")
            summer_total_yield = st.number_input("여름 총 생산량 (kg)", value=3000, step=100, min_value=0)
            summer_price = st.number_input("여름 평균 단가 (원/kg)", value=6000, step=500, min_value=0)
            summer_cost_ratio = st.slider("여름철 경영비 비율 (%)", 10, 70, 30)

            st.markdown("**⛄ 겨울 작기**")
            winter_total_yield = st.number_input("겨울 예상 생산량 (kg)", value=1200, step=100, min_value=0)
            market_price = st.number_input("겨울 예상 단가 (원/kg)", value=18000, step=1000, min_value=0)

        with st.expander("3. 시설 투자비 (입력 단위: 만원)", expanded=False):
            cost_film = st.number_input("피복재 (3년) (만원)", value=200, step=50, min_value=0)
            cost_curtain = st.number_input("보온커튼 (5년) (만원)", value=1500, step=100, min_value=0)
            cost_heater = st.number_input("난방기 (10년) (만원)", value=500, step=100, min_value=0)
            cost_facility = st.number_input("기타 설비 (10년) (만원)", value=300, step=100, min_value=0)

        with st.expander("4. 에너지 설정", expanded=False):
            energy_source = st.selectbox("사용 연료", ["면세유(경유)", "농사용 전기"])
            unit_fuel_cost = st.number_input(
                "연료 단가 (원) [경유: 원/L 가정, 전기: 원/kWh 가정]",
                value=1100 if energy_source == "면세유(경유)" else 50,
                min_value=0
            )
            target_temp = st.slider("목표 온도 (℃)", 10, 20, 15)

            insul_type = st.selectbox(
                "보온 등급(U값)",
                ["비닐 1겹 (U=5.5)", "비닐 2겹 (U=4.5)", "다겹보온커튼 (U=2.0)", "고효율 패키지 (U=1.5)"]
            )

        with st.expander("5. 분석 기간(겨울) 설정", expanded=False):
            start_date = st.date_input("겨울 분석 시작일", value=date(2025, 11, 1))
            end_date = st.date_input("겨울 분석 종료일", value=date(2026, 2, 28))
            if end_date <= start_date:
                st.warning("종료일은 시작일보다 이후여야 합니다.")

        st.write("---")
        submit_btn = st.form_submit_button(
            label="🚜 분석 실행",
            type="primary",
            use_container_width=True
        )

# ------------------------------------------------------------
# 6) 계산 실행
# ------------------------------------------------------------
if not submit_btn:
    st.info("👈 왼쪽 메뉴에서 데이터를 입력하고 ‘분석 실행’ 버튼을 눌러주세요.")
    st.stop()

# -------------------------
# (1) U값 세팅
# -------------------------
u_values = {
    "비닐 1겹 (U=5.5)": 5.5,
    "비닐 2겹 (U=4.5)": 4.5,
    "다겹보온커튼 (U=2.0)": 2.0,
    "고효율 패키지 (U=1.5)": 1.5
}
u_val = u_values[insul_type]

# -------------------------
# (2) 외피면적 계산(곡면 측벽 + 연동 논리 반영)
# -------------------------
geom = calc_surface_area(
    gh_type=gh_type,
    span_count=int(span_count),
    gh_width=float(gh_width),
    gh_length=float(gh_length),
    gh_side_h=float(gh_side_h),
    gh_ridge_h=float(gh_ridge_h),
    wing_d=float(wing_d),
    k_roof_multi=float(k_roof_multi)
)
surface_area = geom["surface_area"]

# -------------------------
# (3) 감가상각비(연간 상각액) 계산
# -------------------------
# 입력 단위: 만원 → 원 변환(*10000)
d1 = cost_film / 3
d2 = cost_curtain / 5
d3 = cost_heater / 10
d4 = cost_facility / 10
depreciation = int((d1 + d2 + d3 + d4) * 10000)

# -------------------------
# (4) 겨울 분석 기간 준비
# -------------------------
dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
n_days = len(dates)

if n_days <= 0:
    st.error("분석 기간이 올바르지 않습니다.")
    st.stop()

# 기상자료 로딩 (근거)
weather_df = None
weather_source = "간이(가정) 기온모드"
if weather_file is not None:
    try:
        weather_df = load_weather_csv(weather_file)
        weather_source = "업로드 CSV(근거자료)"
    except Exception as e:
        st.error(f"기상 CSV 로딩 오류: {e}")
        st.stop()

# 간이모드 파라미터
region_info = REGION_DATA[region_name]
base_t = float(region_info["base"])
amp_t = float(region_info["amp"])

# -------------------------
# (5) 난방 효율/발열량(상대비교용 내부계수)
# -------------------------
# ※ 절대치 정밀모델이 아니라 상대비교 목적(내부 계수는 일관되게 유지)
eff = 0.85 if energy_source == "면세유(경유)" else 0.98
calorific = 8500 if energy_source == "면세유(경유)" else 860

# -------------------------
# (6) 겨울 생산량 분배(단순)
# -------------------------
daily_base_yield = winter_total_yield / max(1, n_days)

# -------------------------
# (7) 루프 계산
# -------------------------
winter_revenue = 0.0
winter_fuel_cost = 0.0
total_heating_hours = 0

daily_rows = []

for i, dt in enumerate(dates):
    # 오늘 Tmin/Tmax 결정
    if weather_df is not None and dt in weather_df.index:
        today_min = float(weather_df.loc[dt, "tmin"])
        today_max = float(weather_df.loc[dt, "tmax"])
    else:
        # 간이모드(랜덤 없음): 계절 사인 + 고정 일교차 10℃
        # 기간 내에서 한파가 중간에 극대(대략)되도록
        day_index = i
        denom = max(1, n_days - 1)
        seasonal_trend = base_t - (amp_t * np.sin(np.pi * day_index / denom))
        today_min = float(seasonal_trend)
        today_max = float(today_min + 10.0)

    daily_heat_load = 0.0
    hours_active = 0

    # 시간별 기온(코사인 근사)
    for hour in range(24):
        hour_rad = (hour - 14) * 2 * np.pi / 24
        current_temp = (today_min + today_max) / 2 + (today_max - today_min) / 2 * np.cos(hour_rad)

        if current_temp < target_temp:
            delta_t = target_temp - current_temp
            # 상대비교용 열부하(단위 엄밀 변환 생략)
            daily_heat_load += surface_area * u_val * delta_t
            hours_active += 1

    needed_fuel = daily_heat_load / (calorific * eff)
    day_cost = needed_fuel * unit_fuel_cost

    winter_fuel_cost += day_cost
    total_heating_hours += hours_active

    # 생산량 계절계수(간단)
    season_factor = 1.0
    if dt.month == 1:
        season_factor = 0.8
    elif dt.month in (11, 2):
        season_factor = 1.1

    daily_yield = daily_base_yield * season_factor
    winter_revenue += daily_yield * market_price

    daily_rows.append({
        "date": dt.date().isoformat(),
        "tmin": today_min,
        "tmax": today_max,
        "heating_hours": hours_active,
        "daily_heat_load": daily_heat_load,
        "daily_fuel_cost_won": float(day_cost),
        "daily_yield_kg": float(daily_yield),
        "daily_revenue_won": float(daily_yield * market_price),
    })

# 정수화
winter_revenue = int(winter_revenue)
winter_fuel_cost = int(winter_fuel_cost)

winter_net_profit = winter_revenue - winter_fuel_cost - depreciation

# -------------------------
# (8) 여름 작기(단순화)
# -------------------------
summer_revenue = int(summer_total_yield * summer_price)
summer_cost = int(summer_revenue * (summer_cost_ratio / 100))
summer_net_profit = summer_revenue - summer_cost

# -------------------------
# (9) 연간 합산
# -------------------------
total_annual_revenue = summer_revenue + winter_revenue
total_annual_profit = summer_net_profit + winter_net_profit

# ------------------------------------------------------------
# 7) 결과 출력
# ------------------------------------------------------------
st.header(f"📊 연간 경영 분석 리포트 ({region_name})")
st.caption(f"기상자료: **{weather_source}** | 겨울 분석기간: {start_date} ~ {end_date} ({n_days}일)")

st.subheader("🏠 0. 온실 형상(외피면적) 요약")
cA, cB, cC, cD = st.columns(4)
cA.metric("총 외피면적(근사)", f"{geom['surface_area']:,.1f} ㎡")
cB.metric("측벽 곡선길이 s", f"{geom['s_side']:.2f} m")
cC.metric("총폭(W_total)", f"{geom['W_total']:.1f} m")
cD.metric("지붕면적(근사)", f"{geom['area_roof']:,.1f} ㎡")
st.caption("※ 곡면 측벽은 ‘방풍벽 길이 d(한쪽 벌어짐)’을 이용해 원호로 근사했습니다(상대비교 목적).")

st.subheader("❄️ 1. 겨울 재배 성적표")
avg_hours = total_heating_hours / max(1, n_days)
st.info(f"💡 난방 가동시간(평균): **하루 {avg_hours:.1f}시간** (24시간 시뮬레이션 기반)")

col1, col2, col3 = st.columns(3)
col1.metric("겨울 매출", f"{winter_revenue/10000:,.0f} 만원")
col2.metric("겨울 비용(난방+상각)", f"{(winter_fuel_cost+depreciation)/10000:,.0f} 만원")
col3.metric(
    "겨울 순이익",
    f"{winter_net_profit/10000:,.0f} 만원",
    delta="흑자" if winter_net_profit > 0 else "적자"
)

st.subheader("📅 2. 연간 총 소득 (여름 + 겨울)")
c1, c2, c3 = st.columns(3)
c1.metric("연간 총 매출", f"{total_annual_revenue/10000:,.0f} 만원")
c2.metric(
    "연간 총 순이익",
    f"{total_annual_profit/10000:,.0f} 만원",
    delta=f"겨울 기여: {winter_net_profit/10000:,.0f} 만원"
)
c3.metric("시설 연간 상각비", f"{depreciation/10000:,.0f} 만원")

st.write("---")
st.subheader("💰 소득 구조 시각화")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.caption("계절별 매출")
    st.bar_chart(
        pd.DataFrame({"계절": ["여름", "겨울"], "매출": [summer_revenue, winter_revenue]}).set_index("계절")
    )
with chart_col2:
    st.caption("비용 구조(단순)")
    st.bar_chart(
        pd.DataFrame(
            {"항목": ["여름경영비", "겨울난방비", "시설상각비"], "금액": [summer_cost, winter_fuel_cost, depreciation]}
        ).set_index("항목")
    )

st.success(
    f"""
**📢 최종 진단(상대비교용):**
- 겨울 재배 순이익: **{int(winter_net_profit/10000):,}만원**
- 연간 총 순이익: **{int(total_annual_profit/10000):,}만원**
"""
)

st.caption(
    "주의: 본 결과는 ‘상대비교/유형화’를 위한 시뮬레이션입니다. "
    "절대 난방비의 정밀 예측이 필요하면(설비용량 설계 등) 단위·효율·환산계수의 정밀 보정이 추가로 필요합니다."
)

# ------------------------------------------------------------
# 8) 결과 다운로드(성과물화 핵심)
# ------------------------------------------------------------
daily_df = pd.DataFrame(daily_rows)

summary = {
    "region": region_name,
    "weather_source": weather_source,
    "winter_start": str(start_date),
    "winter_end": str(end_date),
    "winter_days": n_days,
    "gh_type": gh_type,
    "span_count": int(span_count),
    "L_m": float(gh_length),
    "W_m": float(gh_width),
    "Hs_m": float(gh_side_h),
    "Hr_m": float(gh_ridge_h),
    "d_m(one_side)": float(wing_d),
    "k_roof_multi": float(k_roof_multi),
    "surface_area_m2": float(geom["surface_area"]),
    "u_value": float(u_val),
    "target_temp_C": float(target_temp),
    "energy_source": energy_source,
    "unit_fuel_cost": int(unit_fuel_cost),
    "winter_revenue_won": winter_revenue,
    "winter_fuel_cost_won": winter_fuel_cost,
    "depreciation_won": depreciation,
    "winter_net_profit_won": int(winter_net_profit),
    "summer_revenue_won": int(summer_revenue),
    "summer_cost_won": int(summer_cost),
    "summer_net_profit_won": int(summer_net_profit),
    "annual_revenue_won": int(total_annual_revenue),
    "annual_profit_won": int(total_annual_profit),
}
summary_df = pd.DataFrame([summary])

st.write("---")
st.subheader("📥 결과 파일 다운로드")

st.download_button(
    "결과 요약 CSV 다운로드",
    data=summary_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="figbiz_summary_v1.csv",
    mime="text/csv"
)

st.download_button(
    "일자별 난방 로그 CSV 다운로드",
    data=daily_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="figbiz_daily_log_v1.csv",
    mime="text/csv"
)

with st.expander("일자별 로그 미리보기", expanded=False):
    st.dataframe(daily_df.head(20), use_container_width=True)

# ------------------------------------------------------------
# 9) 실행 안내
# ------------------------------------------------------------
with st.expander("✅ 실행 방법(처음 하시는 경우)", expanded=False):
    st.markdown(
        """
1) 파일을 예: `app.py`로 저장  
2) 터미널에서 아래 실행  
```bash
pip install streamlit pandas numpy
streamlit run app.py
