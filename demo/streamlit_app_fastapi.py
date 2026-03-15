import streamlit as st
import requests
import os

st.set_page_config(
    page_title="InsureAgent",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ InsureAgent")
st.caption("Agentic LLM Insurance Claims Processing · Distilling reasoning from GPT-4o mini → Llama-3.2-1B via LoRA")

# ── API CONFIG ────────────────────────────────────────────────
# Local:          http://localhost:8000
# Docker Compose: http://api:8000
# Production:     set INSUREAGENT_API_URL env var
API_URL = os.getenv("INSUREAGENT_API_URL", "http://localhost:8000")

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Architecture")

    st.markdown("**Teacher Model**")
    st.info("GPT-4o mini\nOpenAI API · Used for training data generation and this demo")

    st.markdown("**Student Model**")
    st.info("Llama-3.2-1B-Instruct\nMeta · Fine-tuned via LoRA (rank-16)\nyuanphd/insureagent-lora-v2")

    st.markdown("**Distillation**")
    st.success("345 agentic traces · 5 epochs\n3.4M / 1.24B trainable params (0.275%)\nLoRA · PEFT · TRL · GRPO")

    st.markdown("**Evaluation (19 held-out cases)**")
    st.markdown("""
| Metric | Teacher | Student |
|---|---|---|
| Verdict | 89.5% | 78.9% |
| Payout | 78.9% | 73.7% |
| Tool seq | 100% | 83.2% |
""")

    st.markdown("**~90% inference cost reduction**")

    st.markdown("---")
    st.markdown("**Sample Policyholders**")
    st.code("P-1001  Premium  active\nP-1002  Basic    active\nP-1004  Premium  lapsed\nP-1009  Premium  active\nP-1011  Basic    active")

    st.markdown("**Claim Types**")
    st.markdown("`storm` `hail` `theft` `fire` `flood` `collision`")

    st.markdown("---")
    st.markdown("[GitHub](https://github.com/yuan-phd/insureagent) · [HuggingFace](https://huggingface.co/yuanphd/insureagent-lora-v2)")

# ── INPUT ─────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    claim_text = st.text_area(
        "Claim Description",
        placeholder="e.g. A hailstorm damaged my car windshield and dented the bonnet.",
        height=100
    )

with col2:
    user_id = st.text_input("Policyholder ID", value="P-1001")
    claimed_amount = st.number_input("Claimed Amount ($)", min_value=0, value=2000, step=100)
    model_choice = st.selectbox(
        "Model",
        options=["teacher", "student"],
        format_func=lambda x: "Teacher (GPT-4o mini)" if x == "teacher" else "Student (Llama-3.2-1B + LoRA)",
    )

run_button = st.button("Process Claim", type="primary", use_container_width=True)

# ── AGENT EXECUTION ───────────────────────────────────────────
if run_button:
    if not claim_text.strip():
        st.warning("Please enter a claim description.")
    else:
        with st.spinner("Processing claim..."):
            try:
                response = requests.post(
                    f"{API_URL}/process_claim",
                    json={
                        "user_id": user_id,
                        "claim_text": claim_text,
                        "claimed_amount": float(claimed_amount),
                        "model": model_choice,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to API at {API_URL}. Is the FastAPI server running?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. Student model inference can take up to 2 minutes.")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e} — {response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        verdict = result["verdict"]
        payout = result["payout"]
        reasoning = result["reasoning"]
        trace = result["trace"]
        latency_ms = result["latency_ms"]
        model_used = result["model_used"]

        st.markdown("---")

        # ── VERDICT BANNER ────────────────────────────────────
        if verdict == "APPROVED":
            st.success(f"✅ **APPROVED** · Payout: **${payout}**")
        elif verdict == "DENIED":
            st.error("❌ **DENIED** · Payout: **$0**")
        else:
            st.warning("⚠️ No verdict reached.")

        if reasoning:
            st.markdown(f"> {reasoning}")

        # ── METADATA ──────────────────────────────────────────
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.caption(f"⏱ Latency: {latency_ms:.0f} ms")
        with meta_col2:
            st.caption(f"🤖 Model: {model_used}")

        # ── REASONING TRACE ───────────────────────────────────
        st.markdown("---")
        st.subheader("Reasoning Trace")

        step = 0
        for msg in trace:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant" and "Verdict:" not in content:
                step += 1
                with st.expander(f"**Step {step} · Thought & Action**", expanded=True):
                    st.markdown(f"```\n{content}\n```")

            elif role == "user" and content.startswith("Observation:"):
                with st.expander(f"**Step {step} · Tool Result**", expanded=True):
                    st.markdown(f"```json\n{content[13:].strip()}\n```")

            elif role == "system" and "REJECTED" in content:
                st.warning("⚠️ Hallucination detected — agent stopped.")