import os
import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 전남 무화과 겨울재배 의사결정지원시스템 (Streamlit)
# - 로그인 잠금(Secrets/ENV)
# - 난방 모델: 간이(14h 고정) / 정밀(24h 일변화)
# - 결과: 겨울/여름/연간 매출·비용·순이익 + 차트 + 근거
# ============================================================

# --- 페이지 설정 (반드시 1회, 최상단) ---
st.set_page_config(page_title="전남 무화과 경영 분석기", layout="wide")

# -----------------------------
# 0) 로그인(잠금) 설정
# -----------------------------
def get_password() -> str | None:
    """
    우선순위:
    1) Streamlit secrets: APP_PASSWORD
    2) 환경변수: APP_PASSWORD
    없으면 None (잠금 비활성)
    """
    pw = None
    try:
        pw = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        pw = None
    if not pw:
        pw = os.getenv("APP_PASSWORD")
    return pw

def login_gate():
    pw = get_password()

    # 비밀번호가 설정되지 않으면 잠금 비활성(연구소 내부 테스트용)
    if not pw:
        st.info("ℹ️ APP_PASSWORD가 설정되어 있지 않아 로그인 없이 실행됩니다. (배포 시 secrets에 비밀번호 설정 권장)")
        return

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return

    st.title("🔒 접근 제한 구역")
    st.markdown("### 과수연구소 관계자 외 접근금지")
    st.write("이 시스템은 허가된 사용자만 이용할 수 있습니다.")
    password_input = st.text_input("비밀번호를 입력하세요", type="password")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("로그인", use_container_width=True):
            if password_input == pw:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다.")
    with c2:
        st.caption("배포(Streamlit Cloud)에서는 Settings → Secrets에 APP_PASSWORD를 등록하세요.")

    st.stop()

login_gate()

# -----------------------------
# 1) 앱 타이틀
# -----------------------------
st.title("🗺️ [전남] 무화과 겨울재배 의사결정지원시스템")
st.markdown("겨울철 투자 분석뿐만 아니라, 여름 작기를 포함한 **연간 총 소득**까지 예측합니다.")
st.divider()

# -----------------------------
# 2) 지역 간이 파라미터(초기값)
#    ※ 향후 기상자료 기반 계수로 교체 가능
# -----------------------------
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

U_VALUES = {
    "비닐 1겹 (U=5.5)": 5.5,
    "비닐 2겹 (U=4.5)": 4.5,
    "다겹보온커튼 (U=2.0)": 2.0,
    "고효율 패키지 (U=1.5)": 1.5,
}

# -----------------------------
# 3) 계산 함수들
# -----------------------------
def greenhouse_surface_area(
    gh_width: float,
    gh_length: float,
    gh_side_h: float,
    gh_ridge_h: float,
    span_count: int,
    gh_type: str,
) -> float:
    """
    외피면적 근사.
    - 지붕/마구리: 동수 반영
    - 측벽: 연동의 경우 내부벽 공유 → 외곽 측벽만(2면) 반영 (span_count 미반영)
    """
    roof_height = gh_ridge_h - gh_side_h
    roof_slope_len = math.sqrt((gh_width / 2) ** 2 + roof_height**2)

    area_roof = 2 * roof_slope_len * gh_length * span_count

    # 연동: 외곽 측벽 2면만 존재한다고 근사
    area_side = 2 * gh_length * gh_side_h

    one_end_wall = (gh_width * gh_side_h) + (0.5 * gh_width * roof_height)
    area_end = one_end_wall * 2 * span_count

    return area_roof + area_side + area_end

def annual_depreciation_won(cost_film: float, cost_curtain: float, cost_heater: float, cost_facility: float) -> int:
    """
    입력 단위: 만원
    """
    d1 = cost_film / 3
    d2 = cost_curtain / 5
    d3 = cost_heater / 10
    d4 = cost_facility / 10
    return int((d1 + d2 + d3 + d4) * 10000)

def simulate_outdoor_min_temp(base_t: float, amp_t: float, day_idx: int, days_total: int, scenario: str) -> float:
    """
    간이 계절변화: sin(반주기) 기반 '최저기온' 근사.
    - scenario:
      * "평년": 기본값
      * "한파(보수적)": 최저기온을 추가로 낮춰 리스크 반영(랜덤 없이 결정적)
    """
    seasonal = base_t - (amp_t * np.sin(np.pi * day_idx / days_total))

    if scenario == "한파(보수적)":
        seasonal -= 3.0  # 보수적 하향(필요 시 조정)
    return seasonal

def diurnal_temp_curve(min_t: float, max_t: float, hour: int) -> float:
    """
    일변화: 코사인 곡선 (최고 14시 가정)
    T(hour) = (min+max)/2 + (max-min)/2 * cos((hour-14)*2π/24)
    """
    omega = 2 * np.pi / 24
    return (min_t + max_t) / 2 + (max_t - min_t) / 2 * np.cos((hour - 14) * omega)

def winter_heating_cost_won(
    surface_area: float,
    u_val: float,
    target_temp: float,
    unit_fuel_cost: float,
    energy_source: str,
    region_base: float,
    region_amp: float,
    start: str = "2025-11-01",
    end: str = "2026-02-28",
    heating_model: str = "정밀(24시간)",
    scenario: str = "평년",
) -> tuple[int, float]:
    """
    반환: (난방비 원, 평균 가온시간(시간/일))
    """
    dates = pd.date_range(start, end)
    days_total = len(dates)

    eff = 0.85 if energy_source == "면세유(경유)" else 0.98
    calorific = 8500 if energy_source == "면세유(경유)" else 860  # 간이값(상대비교 기반)

    total_cost = 0.0
    total_hours = 0.0

    for i, _date in enumerate(dates):
        min_t = simulate_outdoor_min_temp(region_base, region_amp, i, days_total, scenario)

        # max_t는 간이 일교차(고정)로 설정
        # 추후 실측 기반 월별/지역별 일교차로 치환 가능
        max_t = min_t + 10.0

        daily_load = 0.0
        hours_active = 0

        if heating_model == "간이(14시간)":
            # 최저기온 기준 14시간 고정 가정
            delta_t = max(target_temp - min_t, 0.0)
            daily_load = surface_area * u_val * delta_t * 14.0
            hours_active = 14 if delta_t > 0 else 0
        else:
            # 정밀 24시간: 시간별 판단
            for hour in range(24):
                out_t = diurnal_temp_curve(min_t, max_t, hour)
                if out_t < target_temp:
                    delta_t = target_temp - out_t
                    daily_load += surface_area * u_val * delta_t * 1.0
                    hours_active += 1

        needed_fuel = daily_load / (calorific * eff) if (calorific * eff) > 0 else 0
        total_cost += needed_fuel * unit_fuel_cost
        total_hours += hours_active

    avg_hours = total_hours / days_total if days_total > 0 else 0
    return int(total_cost), float(avg_hours)

def winter_revenue_won(winter_total_yield: float, market_price: float, start="2025-11-01", end="2026-02-28") -> int:
    dates = pd.date_range(start, end)
    days = len(dates)
    if days == 0:
        return 0

    daily_base_yield = winter_total_yield / days
    revenue = 0.0
    for d in dates:
        season_factor = 1.0
        if d.month == 1:
            season_factor = 0.8
        elif d.month in (11, 2):
            season_factor = 1.1
        revenue += daily_base_yield * season_factor * market_price
    return int(revenue)

# -----------------------------
# 4) 입력 UI
# -----------------------------
with st.sidebar:
    st.header("📝 데이터 입력")
    st.info("입력 후 맨 아래 버튼을 누르세요.")

    # 0. 지역
    with st.expander("0. 지역 선택", expanded=True):
        region_name = st.selectbox("전남 시·군 선택", list(REGION_DATA.keys()))

    # 1. 온실 규격
    with st.expander("1. 온실 규격", expanded=False):
        gh_type = st.radio("온실 형태", ["단동 (1동)", "연동"], horizontal=True)
        span_count = st.number_input("연동 수", value=1 if gh_type == "단동 (1동)" else 3, step=1, min_value=1)
        gh_width = st.number_input("폭 (m)", value=6.0, step=0.5, min_value=1.0)
        gh_length = st.number_input("길이 (m)", value=50.0, step=1.0, min_value=1.0)
        gh_side_h = st.number_input("측고 (m)", value=2.0, step=0.2, min_value=0.5)
        gh_ridge_h = st.number_input("동고 (m)", value=3.5, step=0.2, min_value=1.0)

        floor_area_m2 = gh_width * gh_length * span_count
        floor_area_py = floor_area_m2 / 3.3
        st.caption(f"바닥면적: {floor_area_m2:,.0f} ㎡ (약 {floor_area_py:,.1f} 평)")

    # 2. 연간 생산 계획
    with st.expander("2. 연간 생산 계획", expanded=False):
        st.markdown("**🌞 여름 작기**")
        summer_total_yield = st.number_input("여름 총 생산량 (kg)", value=3000, step=100, min_value=0)
        summer_price = st.number_input("여름 평균 단가 (원/kg)", value=6000, step=500, min_value=0)
        summer_cost_ratio = st.slider("여름철 경영비 비율 (%)", 10, 80, 30)

        st.markdown("---")
        st.markdown("**⛄ 겨울 작기**")
        winter_total_yield = st.number_input("겨울 예상 생산량 (kg)", value=1200, step=100, min_value=0)
        market_price = st.number_input("겨울 예상 단가 (원/kg)", value=18000, step=1000, min_value=0)

    # 3. 시설투자비
    with st.expander("3. 시설투자비(만원)", expanded=False):
        cost_film = st.number_input("피복비닐 (3년, 만원)", value=200, step=50, min_value=0)
        cost_curtain = st.number_input("보온커튼 (5년, 만원)", value=1500, step=100, min_value=0)
        cost_heater = st.number_input("난방기 (10년, 만원)", value=500, step=100, min_value=0)
        cost_facility = st.number_input("기타 설비 (10년, 만원)", value=300, step=100, min_value=0)

    # 4. 에너지/모델 설정
    with st.expander("4. 에너지·모델 설정", expanded=False):
        energy_source = st.selectbox("사용 연료", ["면세유(경유)", "농사용 전기"])
        unit_fuel_cost = st.number_input(
            "연료 단가 (원)",
            value=1100 if energy_source == "면세유(경유)" else 50,
            min_value=0,
        )
        target_temp = st.slider("목표 온도 (℃)", 8, 22, 15)

        insul_type = st.selectbox("보온 등급", list(U_VALUES.keys()))

        heating_model = st.radio("난방 모델", ["정밀(24시간)", "간이(14시간)"], horizontal=True)
        scenario = st.selectbox("기상 시나리오", ["평년", "한파(보수적)"])

    st.write("---")
    submit_btn = st.button("🚜 연간 분석 실행", type="primary", use_container_width=True)

# -----------------------------
# 5) 계산 및 출력
# -----------------------------
if not submit_btn:
    st.info("👈 왼쪽에서 값을 입력하고 **연간 분석 실행**을 눌러주세요.")
    st.stop()

# A) 공통 계산
u_val = U_VALUES[insul_type]
surface_area = greenhouse_surface_area(gh_width, gh_length, gh_side_h, gh_ridge_h, span_count, gh_type)
depreciation = annual_depreciation_won(cost_film, cost_curtain, cost_heater, cost_facility)

region_info = REGION_DATA[region_name]
base_t = region_info["base"]
amp_t = region_info["amp"]

# B) 겨울
winter_revenue = winter_revenue_won(winter_total_yield, market_price)
winter_fuel_cost, avg_hours = winter_heating_cost_won(
    surface_area=surface_area,
    u_val=u_val,
    target_temp=target_temp,
    unit_fuel_cost=unit_fuel_cost,
    energy_source=energy_source,
    region_base=base_t,
    region_amp=amp_t,
    heating_model=heating_model,
    scenario=scenario,
)
winter_net_profit = winter_revenue - winter_fuel_cost - depreciation

# C) 여름 + 연간
summer_revenue = summer_total_yield * summer_price
summer_cost = summer_revenue * (summer_cost_ratio / 100.0)
summer_net_profit = summer_revenue - summer_cost

total_annual_revenue = summer_revenue + winter_revenue
total_annual_profit = summer_net_profit + winter_net_profit

# -----------------------------
# 6) 결과 출력(UI)
# -----------------------------
st.header(f"📊 연간 경영 분석 리포트 ({region_name})")

st.subheader("🏠 온실/모델 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("바닥면적", f"{floor_area_m2:,.0f} ㎡")
c2.metric("외피면적(근사)", f"{surface_area:,.0f} ㎡")
c3.metric("보온(U)", f"{u_val:.1f}")
c4.metric("난방모델", f"{heating_model} / {scenario}")

st.caption(f"평균 난방 가동시간(추정): **{avg_hours:.1f} 시간/일**")

st.divider()

# 1. 겨울
st.subheader("❄️ 1. 겨울 재배 투자 성적표")
col1, col2, col3 = st.columns(3)
col1.metric("겨울 매출", f"{winter_revenue/10000:,.0f} 만원")
col2.metric("겨울 비용(난방+상각)", f"{(winter_fuel_cost+depreciation)/10000:,.0f} 만원")
col3.metric(
    "겨울 순이익",
    f"{winter_net_profit/10000:,.0f} 만원",
    delta="투자 성공" if winter_net_profit > 0 else "투자 주의",
)

# 2. 연간
st.subheader("📅 2. 연간 총 소득 (여름 + 겨울)")
c1, c2, c3 = st.columns(3)
c1.metric("연간 총 매출", f"{total_annual_revenue/10000:,.0f} 만원")
c2.metric("연간 총 순이익", f"{total_annual_profit/10000:,.0f} 만원")
c3.metric("겨울 기여(순이익)", f"{winter_net_profit/10000:,.0f} 만원")

st.write("---")
st.subheader("💰 소득 구조 시각화")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.caption("계절별 매출 비중")
    df_rev = pd.DataFrame({"계절": ["여름 작기", "겨울 작기"], "매출액": [summer_revenue, winter_revenue]}).set_index("계절")
    st.bar_chart(df_rev)

with chart_col2:
    st.caption("비용 구조 분석")
    df_cost = pd.DataFrame(
        {"항목": ["여름 경영비", "겨울 난방비", "시설 감가상각비"], "금액": [summer_cost, winter_fuel_cost, depreciation]}
    ).set_index("항목")
    st.bar_chart(df_cost)

st.success(
    f"""
**📢 최종 진단**
- 여름 순이익: **{int(summer_net_profit/10000):,}만원**
- 겨울 순이익: **{int(winter_net_profit/10000):,}만원**
- 연간 총 순이익: **{int(total_annual_profit/10000):,}만원**
"""
)

st.write("---")
with st.expander("📚 분석 근거 및 데이터 출처 보기 (Reference)"):
    st.markdown(
        """
### 1) 기상 데이터(현 버전)
- 본 앱의 REGION_DATA(base/amp)는 **간이 비교용 파라미터**입니다.
- 2026년 과제에서는 기상자료(예: 10년치 시간별 기온) 기반으로 지역·월별 계수를 도출하여 고도화합니다.

### 2) 난방부하 산정 개념
- 기본 구조: **외피면적 × U값 × (목표온도 - 외기온)** 의 시간 적분(또는 간이 14시간 가정)
- 외피면적: 지붕 + 측벽(연동은 외곽만) + 마구리(동수 반영)

### 3) 에너지(간이값)
- 면세유: 발열량 8,500 (kcal/L 가정), 효율 85%
- 전기: 열당량 860 (kcal/kWh 가정), 효율 98%
※ 절대값보다 **처리 간 상대비교/의사결정 지원** 목적에 적합

### 4) 감가상각
- 정액법: 피복재(3년), 보온커튼(5년), 난방기/기타(10년)
- 입력 단위: 만원 → 원화 환산 후 연간 상각
"""
    )

# QR (선택)
with st.sidebar:
    st.write("---")
    st.markdown("**📱 모바일로 접속하기(선택)**")
    qr_data = st.text_input("앱 URL(선택)", value="", help="배포 후 Streamlit URL을 넣으면 QR이 생성됩니다.")
    if qr_data.strip():
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={qr_data.strip()}"
        st.image(qr_url, caption="카메라로 스캔하세요")
