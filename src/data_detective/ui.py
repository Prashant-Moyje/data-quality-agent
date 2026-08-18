"""Streamlit front end.

It is a thin client over the HTTP API — it holds no agent logic. That
separation is deliberate: it proves the backend is genuinely reusable (a cron
job or another service could call it), and it keeps the demo honest.

Run with:  streamlit run src/data_detective/ui.py
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

SEVERITY_COLOR = {
    "critical": "#b3261e",
    "high": "#c2610a",
    "medium": "#8a6d00",
    "low": "#3a6b35",
}

st.set_page_config(page_title="Data Detective", page_icon="🔎", layout="wide")
st.title("🔎 Data Detective")
st.caption("An agent that investigates your dataset and tells you what's wrong with it.")

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. Your file is profiled deterministically in Python.\n"
        "2. The agent forms hypotheses about what's broken.\n"
        "3. It writes pandas code and runs it in a sandbox.\n"
        "4. It records evidence-backed findings.\n"
        "5. You get a report and a cleaning script."
    )
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API up · {h.get('model')}")
    except Exception:
        st.error(f"API unreachable at {API_URL}\n\nStart it with:\n`uvicorn data_detective.api:app`")

uploaded = st.file_uploader("Dataset", type=["csv", "parquet", "xlsx"])
context = st.text_area(
    "Context (optional but improves the audit a lot)",
    placeholder="Customer churn data from our CRM. `churned` is the target. "
                "One row per customer. Exported monthly.",
    height=80,
)

if uploaded is not None:
    st.dataframe(pd.read_csv(uploaded, nrows=5) if uploaded.name.endswith(".csv") else None,
                 use_container_width=True)
    uploaded.seek(0)

if st.button("Run audit", type="primary", disabled=uploaded is None):
    with st.status("Starting audit...", expanded=True) as status:
        try:
            resp = requests.post(
                f"{API_URL}/audits",
                files={"file": (uploaded.name, uploaded.getvalue())},
                data={"context": context},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            status.update(label="Failed to start", state="error")
            st.error(str(e))
            st.stop()

        run_id = resp.json()["run_id"]
        report = None
        for _ in range(180):  # up to ~6 minutes
            time.sleep(2)
            try:
                s = requests.get(f"{API_URL}/audits/{run_id}", timeout=10).json()
            except Exception:
                continue
            status.update(label=s.get("progress") or "Working...")
            if s["status"] != "running":
                report = s.get("report")
                break

        if report is None:
            status.update(label="Timed out", state="error")
            st.stop()
        status.update(label="Audit complete", state="complete")

    if report["status"] == "failed":
        st.error(report.get("error", "Unknown failure"))
        st.stop()

    summary = report.get("summary") or {}
    findings = report.get("findings", [])
    usage = report.get("usage", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall risk", str(summary.get("overall_risk", "?")).upper())
    c2.metric("Findings", len(findings))
    c3.metric("Agent steps", report.get("steps_used", 0))
    c4.metric("Tokens", f"{usage.get('input_tokens',0) + usage.get('output_tokens',0):,}")

    if summary:
        st.info(summary.get("summary", ""))
        if not summary.get("ready_for_modeling", True):
            st.warning("The agent does not consider this dataset ready for modeling.")

    tab1, tab2, tab3 = st.tabs(["Findings", "Agent trace", "Cleaning script"])

    with tab1:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for f in sorted(findings, key=lambda x: order.get(x["severity"], 9)):
            color = SEVERITY_COLOR.get(f["severity"], "#555")
            with st.expander(f"[{f['severity'].upper()}] {f['title']}", expanded=f["severity"] in ("critical", "high")):
                st.markdown(
                    f"<span style='background:{color};color:white;padding:2px 8px;"
                    f"border-radius:4px;font-size:12px'>{f['category']}</span> "
                    f"`{'`, `'.join(f.get('columns') or []) or '—'}`",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Evidence.** {f['evidence']}")
                st.markdown(f"**Why it matters.** {f['why_it_matters']}")
                st.markdown(f"**Fix.** {f['recommendation']}")
                if f.get("fix_code"):
                    st.code(f["fix_code"], language="python")

    with tab2:
        st.caption("Every tool call the agent made, in order. This is the audit trail.")
        for t in report.get("trace", []):
            icon = "✅" if t["ok"] else "⚠️"
            label = t["input"].get("hypothesis") or t["input"].get("title") or t["tool"]
            with st.expander(f"{icon} Step {t['step']} · `{t['tool']}` · {str(label)[:80]}"):
                if t["tool"] == "run_pandas":
                    st.code(t["input"].get("code", ""), language="python")
                else:
                    st.json(t["input"])
                st.text(t["output_preview"])

    with tab3:
        script = requests.get(f"{API_URL}/audits/{run_id}/fix_script.py", timeout=10).text
        st.code(script, language="python")
        st.download_button("Download fix_script.py", script, file_name="fix_script.py")
        md = requests.get(f"{API_URL}/audits/{run_id}/report.md", timeout=10).text
        st.download_button("Download report.md", md, file_name="report.md")
