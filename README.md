# 🧴 Skincare Match Tool

An AI-powered web app that analyzes skincare product labels and tells you whether a product is right for your skin. Upload a photo of an ingredient list, select your skin type and concerns, and get a detailed breakdown of every ingredient, with personalized suggestions for better alternatives.

---

## Features

- **OCR ingredient extraction** — uses OpenCV + Tesseract to read ingredient lists from product label photos
- **ML-powered analysis** — a scikit-learn model classifies each ingredient as *Good Fit*, *Possible Irritation*, or *Poor Fit* based on your skin profile
- **Per-ingredient breakdown** — sensitivity score, breadth of benefit, matched concerns, and model confidence
- **Smart suggestions** — recommends the top 3 ingredients that would work better for your skin type and concerns
- **Plain-English summary** — clear overall verdict so you know whether to buy, patch-test, or avoid


---

## Prerequisites

- Python 3.9+
- **Tesseract OCR** installed on your system (required for ingredient text extraction)

### Install Tesseract

**macOS**
```bash
brew install tesseract
```

**Ubuntu / Debian**
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

**Windows**

Download and run the installer from the [Tesseract GitHub releases page](https://github.com/UB-Mannheim/tesseract/wiki). During installation, note the install path (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`) — you may need to add it to your system PATH or set it in your `.env`.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/myadang04/skincare-ai.git
cd skincare-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the `.env` file and fill in any required values (e.g. Tesseract path on Windows):

```bash
# .env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows only; omit on Mac/Linux
```

If you are on macOS or Linux and Tesseract is on your PATH, no changes to `.env` are needed.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. **Select your skin type** (Normal, Oily, Dry, Combination, or Sensitive)
2. **Choose your skin concerns** (Acne, Redness, Hyperpigmentation, etc.)
3. **Upload a photo** of the product's ingredient list (JPG, PNG, or WebP)
4. Click **Analyze Product**

The app will:
- Extract ingredients from the label via OCR
- Run each ingredient through the ML model against your profile
- Show an overall fit verdict, a per-ingredient breakdown, and top suggestions
