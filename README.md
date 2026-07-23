# Multimodal E-Commerce Product Review Analysis Platform 🛍️

An AI-powered, multimodal review analysis platform designed to inspect product reviews and images, giving sellers and buyers a detailed **100-point Product Quality Score**. The platform processes text and image data across five core categories: **Electronics**, **Beauty & Personal Care**, **Home & Kitchen**, **Office Products**, and **Sports & Outdoors**.

---

## 🏗️ Architecture & Core Features

The system evaluates product quality through four core AI dimensions:

1. **Customer Sentiment Analysis (Text)**: A fine-tuned `DistilBERT` classification model that categorizes customer reviews into **Positive**, **Neutral**, or **Negative** sentiments along with confidence scores (e.g. `Positive (91%)`).
2. **Aspect-Based Sentiment Analysis / ABSA (Text)**: A parser that extracts satisfaction scores and sentiments for specific product aspects (e.g., *battery life*, *charging speed*, *durability*, *pricing*).
3. **Review Authenticity (Linguistic Anomaly)**: A hybrid model (`Gradient Boosting` + `Isolation Forest` + `DistilBERT`) that flags suspicious reviews. It parses reviews into three risk levels:
   * 🟢 **Genuine** (`probability < 0.40`)
   * 🟡 **Suspicious** (`0.40 <= probability < 0.70`)
   * 🔴 **Fake** (`probability >= 0.70`)
   * *Includes explanation flags mapping linguistic markers (e.g., lack of detail, template reviews).*
4. **Image Defect & Anomaly Detection (Computer Vision)**: A `ResNet-50` model trained to identify cosmetic or structural anomalies on product images.
5. **Quality Score Fusion**: A regression layer that merges these dimensions into a final `0–100` score using dynamic weights depending on data availability (e.g., if no image is uploaded, defect analysis is gracefully bypassed).

---

## 📁 Directory Structure

```directory
├── app/
│   ├── api.py               # FastAPI backend server & endpoint handlers
│   ├── ui.py                # Streamlit user interface frontend
│   └── database.py          # SQLite persistence schema for history tracking
├── configs/
│   ├── data_config.yaml     # Dataset categories & training configurations
│   └── best_hparams.yaml    # Optuna-optimized model hyperparameters
├── models/                  # Saved model weights & tokenizers
├── src/                     # Core python backend source code
│   ├── aspect/              # Aspect-based sentiment analysis modules
│   ├── defect/              # Computer vision anomaly/defect models
│   ├── fake_reviews/        # Fake review classifiers & feature builders
│   ├── fusion/              # Multimodal regression scoring logic
│   ├── sentiment/           # Sentiment classifier definitions & training
│   └── utils/               # Scraper interfaces, metrics & preprocessors
├── requirements.txt         # Project dependencies
```

---

## 🚀 Quick Start

### 1. Installation
Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment variables
Create a `.env` file in the root directory:
```env
APIFY_TOKEN="your_apify_token_here"
SCRAPERAPI_KEY="your_scraperapi_key_here"
```

### 3. Run the Backend API Server
Start the FastAPI server:
```bash
python app/api.py
```
*The server will run on `http://localhost:8000`.*

### 4. Run the Frontend UI
Start the Streamlit interface:
```bash
streamlit run app/ui.py
```
*The web interface will open automatically in your browser.*

---

## 📊 Datasets & Sources
*   **Customer Sentiment & ABSA**: Amazon Reviews dataset curated by the **McAuley Lab Amazon Reviews 2023**.
*   **Fake Review Detection**: Deceptive/spam review datasets compiled from the **Amazon Product Review (Spam and Not Spam) available on Kaggle** corpus.
*   **Image Defect Detection**: High-resolution industrial anomaly images sourced from the **MVTec AD (Anomaly Detection)** dataset.

---

## 🧪 Training & Hyperparameter Tuning

If you wish to re-train the sentiment classifier on your dataset:

1. **Download Raw Data**:
2. **Preprocess Data**:
   ```bash
   python src/utils/preprocess.py
   ```
3. **Hyperparameter Tuning (Optuna)**:
   Finds the best learning rate, dropout, and batch sizes:
   ```bash
   python src/sentiment/tune.py --n_trials 20
   ```
4. **Train Model**:
   Trains the classifier for 10 epochs using early stopping (patience = 4):
   ```bash
   python src/sentiment/train.py
   ```
