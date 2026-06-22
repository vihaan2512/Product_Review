from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer
from loguru import logger

from src.sentiment.model import SentimentClassifier, load_model
from src.utils.preprocess import TextPreprocessor


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

PREPROCESS_CFG = {
    "text": {"min_word_count": 5, "max_word_count": 300, "max_length": 256},
    "rating": {"positive_threshold": 4, "negative_threshold": 2},
}


class SentimentPredictor:

    def __init__(
        self,
        model_path: str = "models/sentiment_best.pt",
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        confidence_threshold: float = 0.70,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold

        logger.info(f"Loading sentiment predictor from {model_path}...")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python src/sentiment/train.py first."
            )

        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = load_model(model_path, self.device)
        self.preprocessor = TextPreprocessor(PREPROCESS_CFG)

        logger.success("Sentiment predictor ready.")

    def _tokenize(self, text: str) -> dict:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def predict(self, text: str) -> dict:
        
        clean = self.preprocessor.clean_text(text)
        if not clean:
            return {"label": "unknown", "confidence": 0.0, "scores": {}, "uncertain": True}

        encoding = self._tokenize(clean)
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_id    = probs.argmax()
        confidence = float(probs[pred_id])
        label      = ID2LABEL[pred_id]

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "scores": {
                "negative": round(float(probs[0]), 4),
                "neutral":  round(float(probs[1]), 4),
                "positive": round(float(probs[2]), 4),
            },
            "uncertain": confidence < self.confidence_threshold,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            clean_texts = [self.preprocessor.clean_text(t) for t in batch_texts]

            encodings = self.tokenizer(
                clean_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids      = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs  = F.softmax(logits, dim=1).cpu().numpy()

            for prob in probs:
                pred_id    = prob.argmax()
                confidence = float(prob[pred_id])
                results.append({
                    "label":      ID2LABEL[pred_id],
                    "confidence": round(confidence, 4),
                    "scores": {
                        "negative": round(float(prob[0]), 4),
                        "neutral":  round(float(prob[1]), 4),
                        "positive": round(float(prob[2]), 4),
                    },
                    "uncertain": confidence < self.confidence_threshold,
                })

        return results

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:

        texts   = df[text_col].tolist()
        results = self.predict_batch(texts)

        df = df.copy()
        df["pred_label"]      = [r["label"]      for r in results]
        df["pred_confidence"] = [r["confidence"] for r in results]
        df["pred_uncertain"]  = [r["uncertain"]  for r in results]

        uncertain_pct = df["pred_uncertain"].mean() * 100
        logger.info(
            f"Batch prediction complete — "
            f"{len(df)} reviews | "
            f"Uncertain: {uncertain_pct:.1f}%"
        )
        return df

if __name__ == "__main__":
    test_reviews = [
        "This product is absolutely amazing, best purchase I've made all year!",
        "Terrible quality, broke after two days. Complete waste of money.",
        "It's okay I guess, nothing special but does what it says.",
        "ok",   
    ]

    predictor = SentimentPredictor()
    for review in test_reviews:
        result = predictor.predict(review)
        flag = " ⚠ UNCERTAIN" if result["uncertain"] else ""
        print(f"\nReview: {review[:60]}...")
        print(f"  Label:      {result['label']}{flag}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Scores:     {result['scores']}")