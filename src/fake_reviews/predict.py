from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger

from src.fake_reviews.detector import FakeReviewDetector
from src.fake_reviews.features import LinguisticFeatureExtractor
from src.utils.preprocess import TextPreprocessor


PREPROCESS_CFG = {
    "text":   {"min_word_count": 3, "max_word_count": 500, "max_length": 256},
    "rating": {"positive_threshold": 4, "negative_threshold": 2},
}


class FakeReviewPredictor:

    def __init__(
        self,
        model_path: str = "models/fake_review_detector.joblib",
        transformer_dir: str = "models/fake_review_transformer",
        threshold: float = 0.5,
    ):
        self.preprocessor = TextPreprocessor(PREPROCESS_CFG)
        self.linguistic   = LinguisticFeatureExtractor()
        
        self.transformer_dir = Path(transformer_dir)
        has_weights = (self.transformer_dir / "pytorch_model.bin").exists() or (self.transformer_dir / "model.safetensors").exists()
        if self.transformer_dir.exists() and has_weights:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info("Loading fine-tuned Transformer (DistilBERT) for fake review detection...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.transformer_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.transformer_dir)).to(self.device)
            self.model.eval()
            self.use_transformer = True
            self.threshold = 0.5  
            logger.success(f"Transformer model loaded successfully on {self.device.upper()}.")
        else:
            logger.info("Transformer model not found. Falling back to two-stage joblib detector...")
            self.detector     = FakeReviewDetector.load(model_path)
            self.threshold    = getattr(self.detector, "threshold", threshold)
            self.use_transformer = False
            logger.success(f"Fallback joblib detector ready. (Threshold: {self.threshold:.2f})")

    def predict(
        self,
        text: str,
        rating: float = None,
        verified: bool = None,
    ) -> dict:
       
        clean = self.preprocessor.clean_text(text)

        if self.use_transformer:
            import torch
            inputs = {k: torch.tensor(v).to(self.device) for k, v in self.tokenizer([clean], truncation=True, padding=True, max_length=128).items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                score = float(probs[0][1].cpu().numpy())
        else:
            row = {"clean_text": clean}
            if rating   is not None: row["rating"]            = rating
            if verified is not None: row["verified_purchase"] = int(verified)
            df    = pd.DataFrame([row])
            score = self.detector.score(df, text_col="clean_text")[0]

        if score < 0.40:
            is_fake = False
            risk_level = "low"
            classification_label = "Genuine"
        elif score < 0.70:
            is_fake = True
            risk_level = "medium"
            classification_label = "Suspicious"
        else:
            is_fake = True
            risk_level = "high"
            classification_label = "Fake"

        flags = self._generate_flags(clean, score, rating, verified)

        distance   = abs(score - 0.5)
        confidence = "high" if distance > 0.25 else "medium" if distance > 0.1 else "low"

        return {
            "fake_score":  round(float(score), 4),
            "is_fake":     bool(is_fake),
            "risk_level":  risk_level,
            "label":       classification_label,
            "flags":       flags,
            "confidence":  confidence,
        }

    def predict_product_reviews(
        self,
        reviews: list[str],
        ratings: list[float] = None,
        asin: str = "UNKNOWN",
    ) -> pd.DataFrame:
       
        rows = []
        for i, text in enumerate(reviews):
            row = {"clean_text": self.preprocessor.clean_text(text), "asin": asin}
            if ratings: row["rating"] = ratings[i]
            rows.append(row)

        df     = pd.DataFrame(rows)
        result = self.detector.predict(df, text_col="clean_text",
                                       threshold=self.threshold)

        result["original_text"] = reviews
        result = result[["original_text", "fake_score", "is_fake",
                          "risk_level"]].sort_values("fake_score", ascending=False)

        n_fake = result["is_fake"].sum()
        logger.info(
            f"Product {asin}: {len(reviews)} reviews | "
            f"Flagged: {n_fake} ({n_fake/len(reviews)*100:.1f}%)"
        )
        return result.reset_index(drop=True)

    def _generate_flags(
        self,
        text: str,
        score: float,
        rating: float = None,
        verified: bool = None,
    ) -> list:

        flags = []
        feats = self.linguistic.extract(text)

        if feats["generic_phrase_ratio"] > 0.1:
            flags.append("high proportion of generic phrases")
        if feats["type_token_ratio"] < 0.5:
            flags.append("low vocabulary diversity (repetitive)")
        if feats["exclamation_ratio"] > 2:
            flags.append("excessive exclamation marks")
        if feats["caps_ratio"] > 0.15:
            flags.append("unusual capitalization")
        if rating in [1, 5]:
            flags.append("extreme star rating (1 or 5 stars)")
        if verified is False or verified == 0:
            flags.append("unverified purchase")
        if score > 0.7 and not flags:
            flags.append("unusual pattern in embedding space")

        return flags if flags else ["no specific flags — low-level anomaly"]


if __name__ == "__main__":
    from src.utils.logger import setup_logger
    setup_logger()

    predictor = FakeReviewPredictor()

    test_reviews = [
        ("Great product! Highly recommend! Works as described! Amazing quality! "
         "Five stars! Best product ever! Very satisfied! Will buy again!", 5, False),
        ("I bought this for my home office setup. The cable length is perfect "
         "at 6 feet and the connector fits snugly. Build quality feels solid "
         "though the plastic casing is a bit light. Works well with my MacBook Pro. "
         "Shipping took 3 days which was fine.", 4, True),
        ("Good product, does what it says. Happy with the purchase.", 4, True),
    ]

    print("\n" + "="*55)
    print("Fake Review Predictor — Test")
    print("="*55)

    for text, rating, verified in test_reviews:
        result = predictor.predict(text, rating=rating, verified=verified)
        print(f"\nReview: {text[:70]}...")
        print(f"Score:      {result['fake_score']:.3f}")
        print(f"Risk:       {result['risk_level'].upper()}")
        print(f"Is fake:    {result['is_fake']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Flags:      {', '.join(result['flags'])}")