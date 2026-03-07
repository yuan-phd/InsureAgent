import streamlit as st
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from dotenv import load_dotenv
from agent.loop import run_agent

load_dotenv()

st.set_page_config(
    page_title="InsureAgent",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ InsureAgent")
st.caption("Agentic LLM Insurance Claims Processing · Distilling reasoning from GPT-4o mini → Llama-3.2-1B via LoRA")

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Architecture")

    st.markdown("**Teacher Model**")
    st.info("GPT-4o mini\nOpenAI API · Used for training data generation and this demo")

    st.markdown("**Student Model**")
    st.info("Llama-3.2-1B-Instruct\nMeta · Fine-tuned via LoRA (rank-16)\nyuanphd/insureagent-lora-v2")

    st.markdown("**Distillation**")
    st.success("345 agentic traces · 5 epochs\n3.4M / 1.24B trainable params (0.275%)\nLoRA · PEFT · TRL")

    st.markdown("**Evaluation (19 held-out cases)**")
    st.markdown("""
| Metric | Teacher | Student |
|---|---|---|
| Verdict | 89.5% | 80.1% |
| Payout | 78.9% | 73.7% |
| Tool seq | 100% | 63.2% |
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

run_button = st.button("Process Claim", type="primary", use_container_width=True)

# ── AGENT EXECUTION ───────────────────────────────────────────
if run_button:
    if not claim_text.strip():
        st.warning("Please enter a claim description.")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY not found in .env file.")
        else:
            client = OpenAI(api_key=api_key)

            with st.spinner("Processing claim..."):
                try:
                    trace = run_agent(
                        claim_text=claim_text,
                        user_id=user_id,
                        claimed_amount=float(claimed_amount),
                        client=client
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # Extract verdict and payout first
            verdict = None
            payout = None
            reasoning = None
            for msg in reversed(trace):
                if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
                    content = msg["content"]
                    verdict = "APPROVED" if "APPROVED" in content else "DENIED"
                    payout_match = re.search(r'Payout:\s*\$?([\d,]+\.?\d*)', content)
                    payout = payout_match.group(1) if payout_match else "0"
                    reasoning_match = re.search(r'Reasoning:\s*(.+)', content, re.DOTALL)
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                    break

            st.markdown("---")

            # ── VERDICT BANNER (top) ──────────────────────────
            if verdict == "APPROVED":
                st.success(f"✅ **APPROVED** · Payout: **${payout}**")
            elif verdict == "DENIED":
                st.error(f"❌ **DENIED** · Payout: **$0**")
            else:
                st.warning("⚠️ No verdict reached.")

            if reasoning:
                st.markdown(f"> {reasoning}")

            # ── REASONING TRACE (below verdict) ──────────────
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