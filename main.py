import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import json
import os
import requests
from GNN.main import gnnLLM
from cv_fac import process_frame

# ---------- Streamlit Config ----------
st.set_page_config(page_title="🧠 Neurofibro Progressor", layout="wide")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select a Module", [
    "GAIT Module",
    "Disease Progression",
    "Diagnosis",
    "GNN Drug Diagnosis"
])

# ---------- Send to LLaMA ----------
def send_to_llama(gait_data):
    system_prompt = "You are a medical AI expert diagnosing neurological gait disorders based on these metrics..."
    user_prompt = f"Here are the gait metrics:\n{json.dumps(gait_data, indent=2)}\n\nProvide a diagnosis."

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "posgait",
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False
        })
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"❌ LLaMA backend error. Status code: {response.status_code}"
    except Exception as e:
        return f"❌ Failed to connect to LLaMA: {str(e)}"

# ---------- Helper to Average Metrics ----------
def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    return {key: sum(m[key] for m in metrics_list if key in m) / len(metrics_list)
            for key in metrics_list[0] if isinstance(metrics_list[0][key], (int, float))}

# ---------- GAIT Video Processor ----------
class GaitVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.metrics_list = []
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        output, live_metrics = process_frame(image, return_metrics=True)
        st.session_state['latest_metrics'] = live_metrics
        self.metrics_list.append(live_metrics)
        return av.VideoFrame.from_ndarray(output, format="bgr24")

# ---------- GAIT Module ----------
if page == "GAIT Module":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("🚶 GAIT Analysis")
    st.write("Walk normally in front of the webcam and click 'Stop' when done.")

    st.session_state.setdefault('capture_complete', False)
    st.session_state.setdefault('gait_summary', None)
    st.session_state.setdefault('show_diagnosis_button', False)
    st.session_state.setdefault('latest_metrics', {})

    # Start video stream
    ctx = webrtc_streamer(
        key="gait-analyzer",
        video_processor_factory=GaitVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    st.divider()
    st.subheader("📱 Real-time Gait Metrics")
    st.json(st.session_state['latest_metrics'])

    if st.button("🛑 Stop"):
        if ctx and ctx.video_processor:
            gait_summary = average_metrics(ctx.video_processor.metrics_list)
            st.session_state['gait_summary'] = gait_summary
            st.session_state['capture_complete'] = True
            st.session_state['show_diagnosis_button'] = True
            st.success("✅ Gait data captured!")
        else:
            st.error("⚠️ Video processor not available.")

    if st.session_state['capture_complete']:
        st.subheader("📊 Gait Metrics Summary (Final)")
        st.json(st.session_state['gait_summary'])

        # -------- Visualization Section --------
        df = pd.DataFrame(ctx.video_processor.metrics_list)
        summary = st.session_state['gait_summary']
        keys = list(summary.keys())

        st.markdown("### 🧠 Visual Insights from Gait Metrics")

        if st.checkbox("📈 Show Metric Trends Over Time"):
            fig1 = px.line(df[keys[:6]], labels={"value": "Metric Value", "index": "Frame"},
                           title="Trend of Gait Metrics Over Time")
            st.plotly_chart(fig1, use_container_width=True)

        if st.checkbox("🕸️ Show Radar Plot Summary"):
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=list(summary.values()),
                theta=list(summary.keys()),
                fill='toself',
                name='Gait Summary'
            ))
            radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(radar_fig, use_container_width=True)

        if st.checkbox("🔍 Show Metric Correlation Heatmap"):
            fig2, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig2)

        if st.checkbox("📊 Compare to Normal Ranges"):
            normal_values = {k: 1.0 for k in summary}
            comparison_df = pd.DataFrame({
                "Metric": summary.keys(),
                "Patient": summary.values(),
                "Normal": normal_values.values()
            })

            bar_fig = px.bar(comparison_df.melt(id_vars=["Metric"], var_name="Source", value_name="Value"),
                             x="Metric", y="Value", color="Source", barmode="group",
                             title="Gait Metrics vs Expected Norms")
            st.plotly_chart(bar_fig, use_container_width=True)

    if st.session_state['show_diagnosis_button']:
        if st.button("🔍 Proceed to Diagnosis"):
            with st.spinner("Sending data to POSGAIT for diagnosis..."):
                result = send_to_llama(st.session_state['gait_summary'])
                st.subheader("🧠 Diagnosis Result")
                st.code(result)
            st.session_state['show_diagnosis_button'] = False
            st.session_state['capture_complete'] = False


# ---------- GNN Drug Diagnosis ----------
elif page == "GNN Drug Diagnosis":
    st.title("💊 GNN Drug-Side Effect Predictor")
    st.markdown("Use Graph Neural Network to predict drug-related side effects.")

    drug_name = st.text_input("🧪 Enter Drug Name")
    medical_history = st.text_area("🩺 Enter Patient Medical History")

    if st.button("🧠 Predict Side Effects"):
        if drug_name and medical_history:
            with st.spinner("Running GNN inference..."):
                try:
                    result = gnnLLM(drug_name, medical_history)
                    st.success("✅ Prediction Generated")
                    with st.expander("📋 Predicted Side Effects"):
                        if isinstance(result, dict):
                            st.json(result)
                        else:
                            st.code(result)
                except Exception as e:
                    st.error(f"❌ GNN error: {str(e)}")
        else:
            st.warning("⚠️ Please fill out both fields.")

# ---------- Disease Progression ----------
elif page == "Disease Progression":
    st.title("🧬 GAN-Based Disease Progression Viewer")
    st.markdown("Visual timeline of disease progression (generated images).")

    image_folder = "images"
    if os.path.exists(image_folder):
        images = sorted(os.listdir(image_folder))
        if images:
            st.image([f"{image_folder}/{img}" for img in images],
                     caption=images, use_container_width=True)
        else:
            st.info("🖼️ Image folder is empty. Add GAN-generated progression images.")
    else:
        st.warning("📁 Image folder not found. Ensure `images/` exists.")

# ---------- Diagnosis ----------
elif page == "Diagnosis":
    st.title("🧠 Diagnosis from Gait Data")

    if 'gait_summary' in st.session_state:
        with st.expander("📊 Gait Summary"):
            st.json(st.session_state['gait_summary'])

        if st.button("📄 Re-run Diagnosis"):
            with st.spinner("🧠 Generating diagnosis..."):
                result = send_to_llama(st.session_state['gait_summary'])
                st.subheader("🧠 Diagnosis Result")
                st.code(result)
    else:
        st.warning("⚠️ No gait data found. Please run the GAIT Module first.")