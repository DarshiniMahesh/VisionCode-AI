"""
NEXUS AI — Backend Server
Flask app serving both the Code Agent and Image Caption Generator
"""

import os
import re
import json
import base64
import subprocess
import tempfile
import pickle
import time
from io import BytesIO
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ── Optional heavy imports (graceful degradation) ─────────────
try:
    import numpy as np
    from PIL import Image
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠  groq/PIL not installed — caption features limited")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠  TensorFlow not installed — using LLM-only captioning")

# ─────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Config — all from .env ────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
FLASK_DEBUG   = os.getenv("FLASK_DEBUG", "False").lower() == "true"
FLASK_PORT    = int(os.getenv("FLASK_PORT", "5000"))

MODEL_DIR      = Path("models")
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
MODEL_PATH     = MODEL_DIR / "best_model.keras"
CONFIG_PATH    = MODEL_DIR / "config.json"

# ── Global state ─────────────────────────────────────────────
_groq_client   = None
_caption_model = None
_resnet_ext    = None
_tokenizer     = None
_idx_to_word   = None
_max_length    = 34
_model_config  = {}

# ─────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────

def init_groq():
    global _groq_client
    if not GROQ_AVAILABLE:
        print("⚠  Groq package not installed")
        return False
    if not GROQ_API_KEY:
        print("⚠  GROQ_API_KEY not set in .env")
        return False
    try:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        r = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Reply: ok"}],
            max_tokens=5,
        )
        print(f"✅ Groq connected: {r.choices[0].message.content.strip()}")
        return True
    except Exception as e:
        print(f"⚠  Groq init failed: {e}")
        _groq_client = None
        return False


def init_caption_model():
    global _caption_model, _resnet_ext, _tokenizer, _idx_to_word, _max_length, _model_config
    if not TF_AVAILABLE:
        print("⚠  TF not available — skipping model load")
        return False
    if not MODEL_PATH.exists():
        print(f"⚠  Model not found at {MODEL_PATH} — place your trained .keras file there")
        return False
    try:
        print("📦 Loading caption model...")
        _caption_model = load_model(str(MODEL_PATH))
        print("✅ LSTM model loaded")

        rb = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        _resnet_ext = Model(inputs=rb.inputs, outputs=rb.output)
        _resnet_ext.trainable = False
        print("✅ ResNet50 extractor ready")

        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
        _idx_to_word = {v: k for k, v in _tokenizer.word_index.items()}

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                _model_config = json.load(f)
            _max_length = _model_config.get("max_length", 34)

        print(f"✅ Tokenizer loaded — vocab {len(_tokenizer.word_index):,}, max_len {_max_length}")
        return True
    except Exception as e:
        print(f"⚠  Model load error: {e}")
        return False


def _extract_features(img_array):
    arr = preprocess_input(img_array.astype(np.float32))
    return _resnet_ext.predict(np.expand_dims(arr, 0), verbose=0)[0]


def _greedy_caption(feat):
    cap = "startseq"
    for _ in range(_max_length):
        seq  = _tokenizer.texts_to_sequences([cap])[0]
        seq  = pad_sequences([seq], maxlen=_max_length, padding="post").astype(np.int32)
        pred = _caption_model.predict([feat.reshape(1, -1), seq], verbose=0)
        w    = _idx_to_word.get(int(np.argmax(pred[0])), "")
        if not w or w == "endseq":
            break
        cap += " " + w
    return cap.replace("startseq", "").strip()


def _beam_caption(feat, beam_width=3, alpha=0.7):
    beams = [[0.0, 0.0, ["startseq"]]]
    for _ in range(_max_length):
        cands = []
        for ns, lp, seq in beams:
            if seq[-1] == "endseq":
                cands.append([ns, lp, seq])
                continue
            enc  = _tokenizer.texts_to_sequences([" ".join(seq)])[0]
            pad  = pad_sequences([enc], maxlen=_max_length, padding="post").astype(np.int32)
            prob = _caption_model.predict([feat.reshape(1, -1), pad], verbose=0)[0]
            for idx in np.argsort(prob)[-beam_width:]:
                w = _idx_to_word.get(int(idx), "")
                if not w:
                    continue
                nlp = lp + np.log(prob[idx] + 1e-10)
                ns2 = [nlp / (len(seq + [w]) ** alpha), nlp, seq + [w]]
                cands.append(ns2)
        beams = sorted(cands, key=lambda x: x[0], reverse=True)[:beam_width]
        if all(b[2][-1] == "endseq" for b in beams):
            break
    return " ".join(w for w in beams[0][2] if w not in ("startseq", "endseq"))


def _llm_caption(img_b64: str, style: str = "natural") -> str:
    if _groq_client is None:
        return "Caption engine unavailable — check your API key in .env"

    prompts = {
        "natural":  "Describe this image in one clear English sentence. Start with the main subject. No preamble.",
        "detailed": "Describe this image in 2 sentences covering subjects, actions and setting.",
        "concise":  "Describe this image in 5-8 words only.",
        "poetic":   "Write one poetic, evocative sentence describing this image.",
    }

    for attempt in range(3):
        try:
            resp = _groq_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": prompts.get(style, prompts["natural"])},
                    ],
                }],
                max_tokens=100,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip().strip('"\'')
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"Caption error: {str(e)[:80]}"
    return "Caption generation failed — please try again"


# ─────────────────────────────────────────────────────────────
# ROUTES — Pages
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/code")
def code_page():
    return render_template("code.html")

@app.route("/caption")
def caption_page():
    return render_template("caption.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ─────────────────────────────────────────────────────────────
# API — Code Agent
# ─────────────────────────────────────────────────────────────

LANGUAGE_CONFIG = {
    "python":     {"ext": ".py",   "run": ["python3"]},
    "javascript": {"ext": ".js",   "run": ["node"]},
    "java":       {"ext": ".java", "run": None},
    "c":          {"ext": ".c",    "run": None},
    "cpp":        {"ext": ".cpp",  "run": None},
    "bash":       {"ext": ".sh",   "run": ["bash"]},
    "sql":        {"ext": ".sql",  "run": None},
    "html":       {"ext": ".html", "run": None},
    "r":          {"ext": ".R",    "run": ["Rscript"]},
}


@app.route("/api/groq", methods=["POST"])
def groq_api():
    if _groq_client is None:
        return jsonify({"error": "API not initialised — check GROQ_API_KEY in .env"}), 500
    data = request.json
    try:
        resp = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": data.get("system", "You are a helpful assistant.")},
                {"role": "user",   "content": data.get("user", "")},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```[\w]*\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$",       "", raw, flags=re.MULTILINE)
        return jsonify({"content": raw.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/execute", methods=["POST"])
def execute_code():
    data     = request.json
    code     = data.get("code", "")
    language = data.get("language", "python")

    if language == "html":
        return jsonify({"success": True, "output": "(HTML rendered in browser)", "is_html": True})

    cfg = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["python"])
    tmp = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=cfg["ext"], delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            tmp = f.name

        if language == "java":
            m = re.search(r"public\s+class\s+(\w+)", code)
            cls = m.group(1) if m else "Main"
            java_path = f"/tmp/{cls}.java"
            with open(java_path, "w") as f:
                f.write(code)
            cr = subprocess.run(["javac", java_path], capture_output=True, text=True, timeout=30)
            if cr.returncode != 0:
                return jsonify({"success": False, "error": cr.stderr})
            rr = subprocess.run(["java", "-cp", "/tmp", cls], capture_output=True, text=True, timeout=30)
            return jsonify({"success": rr.returncode == 0, "output": rr.stdout.strip(), "error": rr.stderr.strip()})

        elif language in ("c", "cpp"):
            compiler = "gcc" if language == "c" else "g++"
            out = "/tmp/nexus_out"
            cr = subprocess.run([compiler, tmp, "-o", out], capture_output=True, text=True, timeout=30)
            if cr.returncode != 0:
                return jsonify({"success": False, "error": cr.stderr})
            rr = subprocess.run([out], capture_output=True, text=True, timeout=30)
            return jsonify({"success": rr.returncode == 0, "output": rr.stdout.strip(), "error": rr.stderr.strip()})

        elif language == "sql":
            proc = subprocess.run(["sqlite3"], input=code, capture_output=True, text=True, timeout=30)
            return jsonify({"success": True, "output": proc.stdout.strip() or "(Query executed)"})

        else:
            run_cmd = cfg["run"] + [tmp]
            rr = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
            return jsonify({
                "success": rr.returncode == 0,
                "output":  rr.stdout.strip(),
                "error":   rr.stderr.strip(),
            })

    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Execution timed out (30s limit)"})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": f"Runtime not found: {e}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# API — Caption Generator
# ─────────────────────────────────────────────────────────────

@app.route("/api/caption", methods=["POST"])
def caption_api():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file  = request.files["image"]
    style = request.form.get("mode", "natural")

    try:
        img_bytes = file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        buf = BytesIO()
        img_resized = img.copy()
        img_resized.thumbnail((512, 512), Image.LANCZOS)
        img_resized.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        greedy_cap = ""
        beam_cap   = ""

        if TF_AVAILABLE and _caption_model is not None:
            arr       = img_to_array(img.resize((224, 224)))
            feat      = _extract_features(arr)
            greedy_cap = _greedy_caption(feat).capitalize() + "."
            beam_cap   = _beam_caption(feat).capitalize()   + "."

        llm_cap = _llm_caption(img_b64, style)
        if llm_cap and not llm_cap.endswith("."):
            llm_cap += "."

        scores = {
            "bleu4":  str(_model_config.get("bleu4",  "—")),
            "meteor": str(_model_config.get("meteor", "—")),
            "cider":  str(_model_config.get("cider",  "—")),
        }

        return jsonify({
            "greedy":      greedy_cap,
            "beam":        beam_cap,
            "llm":         llm_cap,
            "greedy_conf": 72,
            "beam_conf":   88,
            "llm_conf":    95,
            "scores":      scores,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "groq":    _groq_client is not None,
        "model":   _caption_model is not None,
        "resnet":  _resnet_ext is not None,
        "tf":      TF_AVAILABLE,
        "version": "2.0",
    })


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  NEXUS AI — Server Starting")
    print("═" * 50)

    init_groq()
    init_caption_model()

    print("\n🌐 Server ready at http://localhost:5000")
    print("═" * 50 + "\n")

    app.run(debug=FLASK_DEBUG, host="0.0.0.0", port=FLASK_PORT, use_reloader=False)