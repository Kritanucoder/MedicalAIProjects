# 🏥 Medical AI Projects

> Advancing healthcare through artificial intelligence - ECG analysis, dementia detection, adverse drug reaction classification, and beyond

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Kritanucoder/MedicalAIProjects/issues)

A collection of cutting-edge AI implementations for medical diagnostics and clinical NLP, focusing on cardiovascular health, neurological conditions, and pharmacovigilance. These projects leverage deep learning, reinforcement learning, and large language models to transform healthcare delivery.

---

## 📚 Projects

### 1️⃣ SAC-Guided Adverse Drug Reaction Classification

**Reinforcement learning-powered pipeline for automated ADR detection in medical text**

- 🎯 **Purpose**: Classify adverse drug reactions from clinical narratives and patient reports using state-of-the-art NLP models optimized with Soft Actor-Critic (SAC) reinforcement learning
- 🧠 **Technology**: BERT, Multi-GRU, PubMedBERT, BioBERT, ClinicalBERT with SAC-based fine-tuning
- 📊 **Features**:
  - Multi-model benchmarking across 5 transformer architectures
  - RL-optimized model selection and hyperparameter tuning
  - BERT achieves best performance with 1.6% improvement over baseline
  - End-to-end pipeline from raw text to classification
- 💡 **Use Cases**: Pharmacovigilance, clinical decision support, drug safety monitoring, automated medical record analysis

[📓 View Notebook](https://github.com/Kritanucoder/MedicalAIProjects/blob/main/SAC_Guided_ADR_Classification.ipynb)

---

### 2️⃣ LLM-Based ECG Generator

**Generate synthetic electrocardiogram signals using Large Language Models**

- 🎯 **Purpose**: Create realistic ECG waveforms for training ML models and research
- 🧠 **Technology**: LLM-powered synthesis (MiniLLaMA) with customizable parameters
- 📊 **Features**:
  - Synthetic ECG generation with variable morphology
  - Controllable heart rate and rhythm patterns
  - Data augmentation for medical AI training
- 💡 **Use Cases**: Medical education, algorithm testing, dataset expansion

[📓 View Notebook](https://github.com/Kritanucoder/MedicalAIProjects/blob/main/LLM_based_ECG_Generator.ipynb)

---

### 3️⃣ AI-Based Dementia Detector

**Early detection of dementia using AI-powered risk prediction and analysis**

- 🎯 **Purpose**: Identify cognitive decline patterns and predict dementia risk
- 🧠 **Technology**: Classical ML ensemble (XGBoost, LightGBM, CatBoost, RandomForest, SVC, LogisticRegression, KNN)
- 📊 **Features**:
  - Multi-model ensemble with cross-validation
  - Interactive Gradio interface for real-time predictions
  - Comprehensive preprocessing and feature engineering pipeline
  - High accuracy risk assessment
- 💡 **Use Cases**: Clinical screening, research, early intervention planning, decision support

[📓 View Notebook](https://github.com/Kritanucoder/MedicalAIProjects/blob/main/AI_based_Dementia_Detector.ipynb)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch jupyter transformers huggingface-hub gradio
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Kritanucoder/MedicalAIProjects.git

# Navigate to the project
cd MedicalAIProjects

# Launch Jupyter Notebook
jupyter notebook
```

### Running Individual Projects

```bash
# For SAC-Guided ADR Classification
jupyter notebook SAC_Guided_ADR_Classification.ipynb

# For LLM-Based ECG Generator
jupyter notebook LLM_based_ECG_Generator.ipynb

# For AI-Based Dementia Detector
jupyter notebook AI_based_Dementia_Detector.ipynb
```

---

## 🔬 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | TensorFlow, PyTorch |
| **NLP/Transformers** | BERT, BioBERT, ClinicalBERT, PubMedBERT, Hugging Face |
| **Reinforcement Learning** | Soft Actor-Critic (SAC), Stable-Baselines3 |
| **Classical ML** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Gradio |

---

## 📊 Performance Highlights

- **SAC-Guided ADR Classification**: BERT achieves best performance with 1.6% improvement through RL-optimized fine-tuning
- **ECG Generator**: Produces clinically realistic waveforms with customizable parameters
- **Dementia Detector**: Robust ensemble approach with interactive deployment interface
- **Real-time Analysis**: Fast inference suitable for clinical deployment

---

## 🤝 Contributing

Contributions are welcome! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Additional medical AI models

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📖 Research & Citations

These projects are built upon state-of-the-art research in:

- Transformer-based models for medical NLP and clinical text analysis
- Reinforcement learning for model optimization
- Convolutional Neural Networks for medical signal processing
- Attention-based models for time-series analysis
- Generative models for synthetic medical data
- Ensemble methods for robust prediction
- AI-powered early disease detection and pharmacovigilance

---

## ⚠️ Disclaimer

**For Research and Educational Purposes Only**

These tools are designed for research, education, and algorithm development. They are **NOT** intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

---

## 📧 Contact

**Kritanu Chattopadhyay** - [@Kritanucoder](https://github.com/Kritanucoder)

Project Link: [https://github.com/Kritanucoder/MedicalAIProjects](https://github.com/Kritanucoder/MedicalAIProjects)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- Medical datasets and benchmarks from the research community
- Open-source libraries and frameworks that made this work possible
- Healthcare professionals for domain expertise and validation

---

<div align="center">

⭐ **Star this repo if you find it helpful!**

*Made with ❤️ for advancing healthcare through AI*

</div>
