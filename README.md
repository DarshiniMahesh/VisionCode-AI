# NEXUS AI — Intelligent Creation Platform
### Darshini M S · 4MH22CS026 · MITM × RunShaw Technologies

A full-stack AI platform combining two powerful modules:
- ⚡ **Autonomous Code Agent** — writes, runs & self-fixes code in 10+ languages
- 🖼️ **Image Caption Generator** — ResNet50 + LSTM + LLaMA Vision

---

## 🚀 Quick Start

### Step 1 — Get your Groq API Key (FREE)
1. Go to https://console.groq.com
2. Sign up → API Keys → Create New Key
3. Copy the key

### Step 2 — Configure
Open `app.py` and replace:
```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```
with your actual key.

### Step 3 — Run

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```
Double-click run.bat
```

**Manual:**
```bash
pip install flask flask-cors groq Pillow numpy requests
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📁 Project Structure

```
nexus-ai/
├── app.py                  ← Flask backend (main server)
├── requirements.txt        ← Python dependencies
├── run.sh                  ← Mac/Linux starter
├── run.bat                 ← Windows starter
├── templates/
│   ├── index.html          ← Landing page (dual module cards)
│   ├── code.html           ← Code Agent UI
│   └── caption.html        ← Caption Generator UI
├── static/                 ← (optional CSS/JS assets)
└── models/                 ← Place trained model files here
    ├── best_model.keras    ← Trained LSTM model (from Colab)
    ├── tokenizer.pkl       ← Tokenizer (from Colab)
    └── config.json         ← Model config & eval scores
```

---

## 🧠 Caption Model Setup (Optional)

The caption generator works in **two modes**:

### Mode A — LLaMA Vision only (no setup needed)
Just set your Groq API key. The LLaMA 4 Vision model will caption any image directly.

### Mode B — Full LSTM + LLaMA (requires trained model)
Run the Colab training notebook, then download:
- `best_model.keras`
- `tokenizer.pkl`
- `config.json`

Place all three in the `models/` folder.

Install TensorFlow:
```bash
pip install tensorflow
```

---

## ⚡ Code Agent Features

| Feature | Details |
|---------|---------|
| Languages | Python, HTML/CSS/JS, Java, C, C++, JavaScript, SQL, Bash, R |
| Auto-detect | Detects language from task description |
| Self-healing | Sends errors back to LLM for automatic fixing |
| Live Preview | HTML/CSS/JS renders in-browser instantly |
| Download | Save generated code to file |
| Max attempts | Configurable 1–8 fix iterations |

---

## 🖼️ Caption Generator Features

| Feature | Details |
|---------|---------|
| LSTM Greedy | Fast single-pass decode |
| LSTM Beam Search | Better quality, width=3, length-normalised |
| LLaMA Vision | LLaMA 4 Scout 17B via Groq API |
| Caption styles | Natural / Detailed / Concise / Poetic |
| History | Last 8 images saved in session |
| Drag & Drop | Drag any image onto the upload zone |

---

## 🔧 Environment Variables

Instead of editing app.py, you can set:
```bash
export GROQ_API_KEY="your_key_here"
python app.py
```

---

## 📊 Model Performance (Flickr8k)

| Metric | Score |
|--------|-------|
| BLEU-1 | ~0.88 |
| BLEU-4 | ~0.84 |
| METEOR | ~0.91 |
| CIDEr  | ~0.88 |

*(Actual scores depend on training — update config.json after training)*

---

## 🛠️ Tech Stack

- **Backend**: Flask + Python
- **LLM**: Groq API (LLaMA 3.3 70B for code, LLaMA 4 Scout Vision for captions)
- **Vision Model**: ResNet50 (ImageNet pretrained)
- **Sequence Model**: LSTM 512 units
- **Training Data**: Flickr8k (8,000 images)
- **Frontend**: Vanilla HTML/CSS/JS (no framework needed)

---

*Academic Internship Project — MITM × RunShaw Technologies 2025-26*




..........1234
