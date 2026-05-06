#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# NEXUS AI — Quick Start Script (Mac / Linux)
# Run:  chmod +x run.sh && ./run.sh
# ─────────────────────────────────────────────────────────────
set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║        NEXUS AI — Quick Start        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install from https://python.org"; exit 1
fi
echo "✅ Python: $(python3 --version)"

# Virtualenv
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate

# Install deps
echo "📦 Installing dependencies..."
pip install --upgrade pip -q
pip install flask flask-cors groq Pillow numpy requests -q

# Create models dir
mkdir -p models

# Check API key
if grep -q "YOUR_GROQ_API_KEY_HERE" app.py; then
  echo ""
  echo "⚠️  IMPORTANT: Set your Groq API key!"
  echo "   Edit app.py and replace YOUR_GROQ_API_KEY_HERE"
  echo "   Get a free key at: https://console.groq.com"
  echo ""
fi

echo ""
echo "🚀 Starting NEXUS AI server..."
echo "   Open http://localhost:5000 in your browser"
echo ""

python3 app.py
