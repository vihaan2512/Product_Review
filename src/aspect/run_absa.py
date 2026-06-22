from pathlib import Path
from typing import Union

import torch
import spacy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger

from src.utils.logger import setup_logger


# ── Per-category aspect presets ───────────────────────────────

CATEGORY_ASPECTS = {
    "Electronics & Gadgets": [
        "battery life", "camera", "display", "sound quality",
        "build quality", "performance", "price", "connectivity",
        "charging", "comfort",
    ],
    "Clothing & Fashion": [
        "fabric", "fit", "comfort", "quality", "stitching",
        "color", "design", "durability", "price", "sizing",
    ],
    "Food & Beverages": [
        "taste", "flavor", "freshness", "quality", "packaging",
        "value", "quantity", "sweetness", "ingredients", "smell",
    ],
    "Furniture & Home": [
        "build quality", "assembly", "comfort", "durability", "design",
        "material", "size", "price", "stability", "finish",
    ],
    "Beauty & Personal Care": [
        "scent", "texture", "effectiveness", "packaging", "price",
        "skin", "moisturizing", "quality", "ingredients", "durability",
    ],
    "Appliances & Tools": [
        "performance", "build quality", "noise", "efficiency",
        "durability", "ease of use", "features", "price", "power", "design",
    ],
    "Auto (detect from reviews)": [],  
}

ELECTRONICS_ASPECTS = CATEGORY_ASPECTS["Electronics & Gadgets"]

ID2SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}

class ABSAPredictor:

    def __init__(
        self,
        model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading ABSA model: {model_name}")
        logger.info("(First run downloads ~900MB — cached after that)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy loaded for aspect extraction")
        except OSError:
            logger.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm\n"
                "Auto aspect extraction disabled — use explicit aspects instead."
            )
            self.nlp = None

        logger.success("ABSA predictor ready.")

    def _predict_aspects_batched(
        self, text: str, aspects: list[str], batch_size: int = 32
    ) -> dict:
        
        results = {}
        if not aspects:
            return results

        for batch_start in range(0, len(aspects), batch_size):
            batch_aspects = aspects[batch_start: batch_start + batch_size]

            inputs = self.tokenizer(
                [text] * len(batch_aspects),   
                batch_aspects,
                max_length=256,                 
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits         
                probs  = torch.softmax(logits, dim=1).cpu().numpy()  

            for i, aspect in enumerate(batch_aspects):
                p     = probs[i]
                pid   = int(p.argmax())
                sent  = ID2SENTIMENT[pid]
                conf  = float(p[pid])
                
                text_lower = text.lower()
                if aspect.lower() in ["display", "screen"]:
                    display_complaints = ["too bright", "too much bright", "very bright", "glare", "reflective", "harsh", "hurt", "hurts my eyes"]
                    if any(c in text_lower for c in display_complaints):
                        sent = "negative"
                        conf = 0.95
                
                signed = conf if sent == "positive" else (-conf if sent == "negative" else 0.0)
                results[aspect] = {
                    "sentiment":  sent,
                    "confidence": round(conf, 4),
                    "score":      round(signed, 4),
                    "probs": {
                        "negative": round(float(p[0]) if sent != "negative" else 0.95, 4),
                        "neutral":  round(float(p[1]) if sent != "negative" else 0.02, 4),
                        "positive": round(float(p[2]) if sent != "negative" else 0.03, 4),
                    }
                }

        return results

    def extract_aspects_from_text(self, texts: list[str], max_aspects: int = 15) -> list[str]:

        if self.nlp is None:
            return ELECTRONICS_ASPECTS

        candidates = set()
        for text in texts:
            doc = self.nlp(text.lower())
            for chunk in doc.noun_chunks:
                if 1 <= len(chunk.text.split()) <= 3 and len(chunk.text) > 3:
                    candidates.add(chunk.text.strip())
            for token in doc:
                if token.pos_ == "NOUN" and len(token.text) > 3:
                    candidates.add(token.text.strip())

        stopwords = {
            "thing", "product", "item", "lot", "bit", "kind", "type", "way", "overall", "best", "worst",
            "good", "bad", "great", "excellent", "terrible", "performance", "performer", "aspect", "quality",
            "one", "something", "anything", "nothing", "everything"
        }

        candidates = {c for c in candidates if not any(w in c.split() for w in stopwords)}

        candidates = list(candidates)[:max_aspects]

        logger.info(f"Auto-extracted {len(candidates)} aspect candidates from reviews")
        return candidates if candidates else ELECTRONICS_ASPECTS

    def predict(
        self,
        text: str,
        aspects: list[str] = None,
        category: str = "Auto (detect from reviews)",
        min_confidence: float = 0.5,
    ) -> dict:
       
        if aspects is None:
            preset = CATEGORY_ASPECTS.get(category, [])
            if preset:
                aspects = preset
            elif self.nlp is not None:
                aspects = self.extract_aspects_from_text([text])
            else:
                aspects = ELECTRONICS_ASPECTS

        text_lower = text.lower()
        relevant = [a for a in aspects if a.lower() in text_lower]

        if not relevant:
            return {}

        all_results = self._predict_aspects_batched(text, relevant)

        return {
            asp: res for asp, res in all_results.items()
            if res["confidence"] >= min_confidence
        }

    def predict_batch(self, reviews: list[str], aspects: list[str] = None,
                      category: str = "Auto (detect from reviews)") -> list[dict]:

        return [self.predict(text, aspects=aspects, category=category) for text in reviews]

    def aggregate_across_reviews(self, reviews: list[str],
                                 aspects: list[str] = None,
                                 category: str = "Auto (detect from reviews)") -> dict:
       
        from collections import defaultdict
        import numpy as np

        resolved_aspects = aspects
        if resolved_aspects is None:
            preset = CATEGORY_ASPECTS.get(category, [])
            if preset:
                resolved_aspects = preset
            elif self.nlp is not None:
                resolved_aspects = self.extract_aspects_from_text(reviews)
            else:
                resolved_aspects = ELECTRONICS_ASPECTS

        aspect_scores = defaultdict(list)
        for review in reviews:
            result = self.predict(review, aspects=resolved_aspects, category=category)
            for aspect, data in result.items():
                aspect_scores[aspect].append(data["score"])

        aggregated = {}
        for aspect, scores in aspect_scores.items():
            if len(scores) >= 1:      
                aggregated[aspect] = {
                    "mean_score":   round(float(np.mean(scores)), 4),
                    "review_count": len(scores),
                    "sentiment":    "positive" if np.mean(scores) > 0.1
                                    else "negative" if np.mean(scores) < -0.1
                                    else "neutral",
                }

        aggregated = dict(
            sorted(aggregated.items(),
                   key=lambda x: abs(x[1]["mean_score"]),
                   reverse=True)
        )
        return aggregated


# ── Quick test when run directly ─────────────────────────────
if __name__ == "__main__":
    setup_logger()

    test_reviews = [
        "The fabric is really soft and the fit is perfect. "
        "Color faded a bit after washing but overall great value.",

        "Battery life is absolutely terrible, drains in 3 hours. "
        "However the camera quality is stunning. Build quality feels premium.",
    ]

    predictor = ABSAPredictor()

    for category in ["Clothing & Fashion", "Electronics & Gadgets", "Auto (detect from reviews)"]:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")
        for i, review in enumerate(test_reviews[:1], 1):
            results = predictor.predict(review, category=category)
            for aspect, data in results.items():
                bar = "+" * int(abs(data["score"]) * 10)
                sign = "+" if data["score"] > 0 else "-"
                print(f"  {aspect:<20} [{sign}{bar:<10}] {data['sentiment']:<10} ({data['confidence']:.2%})")