# 🧠 Neurofibro Progressor

## 🧬 Project Summary

**Neurofibro Progressor** is a groundbreaking AI diagnostic system that uses **gait patterns**, **posture**, and **medical history** to detect and simulate the progression of **neurological** and **dermatological** disorders — all at a **cost as low as ₹15**, compared to traditional diagnostics priced ₹500–₹4000.

This multimodal system enables early-stage diagnosis using non-invasive cues like walking style and visual posture, enhanced with symptom-aware simulation and clinical-grade precision, making healthcare **accessible, interpretable, and affordable** for all.

---

## 🔍 Key Features

| Module | Description |
|--------|-------------|
| 🧍‍♂️ **Gait Analysis (CV)** | Real-time webcam-based gait extraction using MediaPipe, with key gait metrics (stride length, step width, sway, etc.) |
| 📊 **Graph Neural Network** | PDE-based Disease-Specific Graph Convolution (DSGC) that captures temporal symptom dependencies |
| 🧠 **LLM Diagnosis Engine** | LLaMA-3 powered evaluation fused with clinical PDFs for interpretable AI-backed medical conclusions |
| 🧪 **Skin Condition Progressor** | StyleGAN2-powered GAN module with spectral stability constraints for simulating visual disease progression |
| ⚙️ **Symptom Modeling** | SCLMA (Symptom-Condition Latent Manifold Alignment) for latent space interpolation between disease stages |

---

## 🧠 Novel Contributions

### ✅ 1. Symptom-Condition Latent Manifold Alignment (SCLMA)
A novel manifold learning approach that warps latent GAN spaces using clinical symptom gradients. Enables interpretable interpolation between disease states (e.g., Stage 1 → Stage 3).

### ✅ 2. Disease-Specific Graph Convolution (DSGC)
A temporal Graph Neural Network using PDE-driven dynamics to model evolving symptoms (e.g., tremor onset followed by delayed bradykinesia).

### ✅ 3. Spectral Stability Criterion
A GAN spectral loss that ensures simulated diseases (like tremors) remain **biologically plausible** and do not exhibit unrealistic jitter or frequency noise.

---

## 📈 Dataset Overview

| Dataset Module | Source | Size |
|----------------|--------|------|
| Gait Data | Webcam (custom recorded) | ~1,000+ JSON frame sequences |
| Pose Metrics | Extracted using MediaPipe | ~5 MB per session |
| Clinical PDFs | Research PDFs + Custom Annotation | ~12 PDFs (~8 MB) |
| Disease Images | Custom + FFHQ + Dermatology Dataset | ~20,000 samples |
| GAN Fine-tuning | StyleGAN2-ADA FFHQ + annotated NF1 samples | ~5,000 faces |

---

## 🚀 Tech Stack

- **Frontend:** Streamlit
- **Backend:** PyTorch, Transformers, SentenceTransformers
- **GAN Model:** StyleGAN2-ADA
- **LLM Engine:** LLaMA 3 (via Groq API)
- **Vision:** MediaPipe (Gait + Pose Extraction)
- **Symptom Graphs:** PyTorch Geometric
- **PDF RAG:** LangChain + FAISS VectorDB
- **Dockerized Deployment:** Adaptive RAG with Streamlit

---

## 💻 Local Setup Instructions

```bash
# Clone repo
git clone https://github.com/yourusername/neurofibro-progressor.git
cd neurofibro-progressor

# Create and activate environment
conda create -n neurofibro python=3.10 -y
conda activate neurofibro

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
📦 Repository Structure

neurofibro-progressor/
│
├── app.py                        # Main Streamlit app
├── gait_module/
│   └── cv_fac.py                # Gait extractor using MediaPipe
│
├── diagnosis_module/
│   ├── rag_llm.py               # RAG pipeline using LLaMA
│   └── graph_model.py           # DSGC implementation
│
├── progression_module/
│   └── neurofibro_progressor.py # GAN progression simulation
│
├── data/
│   └── clinical_pdfs/           # Annotated PDFs for RAG
│
├── pretrained/
│   └── stylegan2.pkl            # Pretrained weights
│
└── README.md

🧑‍🚀 Team Valorant Loadout

    🎯 S Kunal Achintya Reddy (Team Lead, Duelist - Took initiative & built core math SCLMA/DSGC)

    🛡️ Kota VKS Prasad (Initiator - GNN Architect & Backend Dev)

    🧠 Nithilesh Bollena (Controller - CV + GAN Tuner)

    🗺️ Sriman Vashishta V (Sentinel - RAG + LangChain Engineer)

🏁 Impact

    📉 Reduced diagnosis cost from ₹500–₹4000 → ₹15–₹40

    📦 Completely local, privacy-friendly diagnostic AI

    📡 Non-invasive, real-time multimodal inputs (webcam & history)

    🎯 Designed to scale across rural and underserved regions

📬 Contact

Have questions or want to collaborate?

📧 kunalachintya@gmail.com
📮 LinkedIn: S Kunal Achintya Reddy

    “Walking into the future of diagnosis—literally.” 👣
