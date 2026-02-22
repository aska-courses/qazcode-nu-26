"""
Streamlit UI – Medical Diagnosis Assistant
Run: streamlit run src/app.py
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://backend:8080/diagnose")

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Медицинский ассистент | QazCode 2026",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Медицинский ассистент диагностики")
st.caption("Введите симптомы пациента на русском языке — система предложит вероятные диагнозы с кодами МКБ-10.")

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")
    st.divider()
    st.markdown("**Примеры симптомов:**")
    examples = [
        "Высокая температура 39°C, кашель, одышка, боль в груди",
        "Тошнота, рвота, боль в правом подреберье, желтуха",
        "Головная боль, головокружение, повышенное давление 160/100",
        "Сыпь на коже, зуд, отёк Квинке после укуса насекомого",
    ]
    for ex in examples:
        if st.button(ex[:55] + "…", use_container_width=True):
            st.session_state["query_input"] = ex

# ── main form ──────────────────────────────────────────────────────────────────
query = st.text_area(
    "Симптомы пациента",
    value=st.session_state.get("query_input", ""),
    placeholder="Например: высокая температура, кашель, боль в груди…",
    height=120,
    key="query_area",
)

diagnose_btn = st.button("🔍 Диагностировать", type="primary", use_container_width=True)

# ── call API & display results ─────────────────────────────────────────────────
if diagnose_btn and query.strip():
    with st.spinner("Анализирую симптомы…"):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                API_URL,
                json={"symptoms": query},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Не удаётся подключиться к серверу. Убедитесь, что сервер запущен.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Ошибка сервера: {e}")
            st.stop()
        elapsed = round(time.perf_counter() - t0, 2)

    diagnoses = data.get("diagnoses", [])

    if not diagnoses:
        st.warning("Диагнозы не найдены. Попробуйте уточнить симптомы.")
        st.stop()

    st.success(f"✅ Найдено {len(diagnoses)} диагноза(-ов) за {elapsed} сек.")
    st.divider()

    # rank colours
    rank_colors = {1: "#d4edda", 2: "#fff3cd", 3: "#f8d7da"}

    for diag in diagnoses:
        rank   = diag["rank"]
        color  = rank_colors.get(rank, "#e2e3e5")
        medal  = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

        with st.container():
            st.markdown(
                f"""
                <div style="background:{color}; border-radius:10px; padding:16px; margin-bottom:12px;">
                  <h4 style="margin:0">{medal} {diag['diagnosis']}</h4>
                  <code style="font-size:0.9em">МКБ-10: {diag['icd10_code']}</code>
                  <p style="margin-top:10px; margin-bottom:0">{diag['explanation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("📋 Сырой JSON-ответ"):
        st.json(data)

elif diagnose_btn:
    st.warning("Пожалуйста, введите симптомы.")
