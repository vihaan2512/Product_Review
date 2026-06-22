from pathlib import Path
from typing import Optional, Union
import time
import concurrent.futures

import torch
import numpy as np
import pandas as pd
from PIL import Image
from loguru import logger

from src.utils.logger import setup_logger
from src.fusion.quality_score import QualityScoreFusion


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(
            f"GPU detected: {torch.cuda.get_device_name(0)} | "
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        logger.warning(
            "No GPU detected — running on CPU. "
            "Sentiment inference will be slow for large review sets. "
            "Consider batching or using Google Colab for demos."
        )
    return device


class AnalysisPipeline:
    
    def __init__(self, weights: dict = None):
        self.device  = get_device()
        self.fusion  = QualityScoreFusion(weights=weights)

        self.batch_size = 64 if self.device.type == "cuda" else 16

        self._sentiment = None
        self._defect    = None
        self._fake      = None
        self._absa      = None

        logger.info(
            f"AnalysisPipeline ready | "
            f"device: {self.device} | "
            f"batch_size: {self.batch_size}"
        )

    @property
    def sentiment(self):
        if self._sentiment is None:
            from src.sentiment.predict import SentimentPredictor
            self._sentiment = SentimentPredictor()
        return self._sentiment

    @property
    def defect(self):
        if self._defect is None:
            from src.defect.predict import DefectPredictor
            self._defect = DefectPredictor()
        return self._defect

    @property
    def fake(self):
        if self._fake is None:
            from src.fake_reviews.predict import FakeReviewPredictor
            self._fake = FakeReviewPredictor()
        return self._fake

    @property
    def absa(self):
        if self._absa is None:
            from src.aspect.run_absa import ABSAPredictor
            self._absa = ABSAPredictor()
        return self._absa
    

    def _log_gpu_memory(self, label: str) -> None:
        if self.device.type == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.debug(f"GPU memory [{label}]: {used:.2f}/{total:.1f} GB")


    def analyze(
        self,
        reviews:          list[str],
        ratings:          list[float]               = None,
        image_path:       Union[str, Path, Image.Image] = None,
        asin:             str                        = "UNKNOWN",
        product_name:     str                        = "Product",
        run_modules:      list[str]                  = None,
        generate_heatmap: bool                       = True,
    ) -> dict:
        
        if not reviews:
            raise ValueError("reviews cannot be empty")

        start_time  = time.time()
        run_modules = run_modules or ["sentiment", "defect", "fake", "absa"]
        results     = {"asin": asin, "product_name": product_name}
        module_outputs = {}
        timing         = {}

        logger.info(
            f"Analyzing {len(reviews)} reviews for {product_name} "
            f"({'with image' if image_path else 'no image — defect excluded'})"
        )

        ood_reviews = []
        for r in reviews:
            if not isinstance(r, str) or len(r.strip()) < 5:
                ood_reviews.append(r)
            elif len([c for c in r if c.isalpha()]) / max(len(r), 1) < 0.3:
                ood_reviews.append(r)
        
        ood_detected = len(ood_reviews) / len(reviews) > 0.5

        fake_results     = []
        sentiment_results = []
        absa_results     = None

        def _run_fake():
            if "fake" not in run_modules:
                return []
            t0 = time.time()
            try:
                logger.info(f"M3 Fake: {len(reviews)} reviews...")
                out = []
                for i, review in enumerate(reviews):
                    rating = ratings[i] if ratings and i < len(ratings) else None
                    out.append(self.fake.predict(review, rating=rating))
                timing["fake"] = round(time.time() - t0, 2)
                n_flagged = sum(1 for r in out if r.get("is_fake"))
                logger.success(f"M3 done in {timing['fake']}s | flagged: {n_flagged}/{len(reviews)}")
                return out
            except Exception as e:
                logger.error(f"M3 Fake detection failed: {e}")
                return []

        def _run_sentiment(clean_revs):
            if "sentiment" not in run_modules:
                return []
            t0 = time.time()
            try:
                logger.info(f"M1 Sentiment: {len(clean_revs)} clean reviews...")
                raw = self.sentiment.predict_batch(clean_revs)
                out = []
                for i, r in enumerate(raw):
                    r_copy = dict(r)
                    r_copy["text"] = clean_revs[i]
                    out.append(r_copy)
                timing["sentiment"] = round(time.time() - t0, 2)
                logger.success(f"M1 done in {timing['sentiment']}s ({len(clean_revs)/timing['sentiment']:.1f} rev/s)")
                return out
            except Exception as e:
                logger.error(f"M1 Sentiment failed: {e}")
                return []

        def _run_absa(clean_revs, category, custom_aspects):
            if "absa" not in run_modules:
                return None
            t0 = time.time()
            try:
                logger.info(f"M4 ABSA: {len(clean_revs)} clean reviews, category={category}...")
                out = self.absa.aggregate_across_reviews(
                    clean_revs,
                    aspects=custom_aspects if custom_aspects else None,
                    category=category or "Auto (detect from reviews)",
                )
                timing["absa"] = round(time.time() - t0, 2)
                logger.success(f"M4 done in {timing['absa']}s | aspects found: {len(out)}")
                return out
            except Exception as e:
                logger.error(f"M4 ABSA failed: {e}")
                return None

        fake_results = _run_fake()
        module_outputs["fake"] = fake_results if fake_results else None

        clean_indices = (
            [i for i, r in enumerate(fake_results) if r.get("fake_score", 0.0) < 0.70]
            if fake_results else list(range(len(reviews)))
        )
        if not clean_indices:
            clean_indices = list(range(len(reviews)))
        clean_reviews = [reviews[i] for i in clean_indices]

        absa_category     = getattr(self, "_absa_category", "Auto (detect from reviews)")
        absa_custom       = getattr(self, "_absa_custom_aspects", None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_sentiment = executor.submit(_run_sentiment, clean_reviews)
            future_absa      = executor.submit(_run_absa, clean_reviews, absa_category, absa_custom)
            sentiment_results = future_sentiment.result()
            absa_results      = future_absa.result()

        module_outputs["sentiment"] = sentiment_results if sentiment_results else None
        module_outputs["absa"]      = absa_results
        self._log_gpu_memory("post-text-modules")

        if "defect" in run_modules:
            if image_path is not None:
                t0 = time.time()
                try:
                    defect_keywords = {
                        "broken", "damaged", "shattered", "cracked", "scratched", "defective",
                        "stopped working", "torn", "ripped", "faulty", "dead", "leak",
                        "leaking", "bent", "dent", "dented", "chipped", "malfunction", 
                        "melted", "burned", "scuff", "scuffs", "stain", "stains"
                    }
                    negations = {"not", "no", "never", "isnt", "arent", "wasnt", "didnt", "dont", "without", "neither"}
                    
                    has_defect_complaint = False
                    for r in reviews:
                        clean_text = r.lower().replace("n't", "nt").replace("'", "")
                        sentences = [s.strip() for s in clean_text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
                        
                        for sentence in sentences:
                            words = sentence.split()
                            for idx, word in enumerate(words):
                                if word in defect_keywords or (idx < len(words)-1 and f"{word} {words[idx+1]}" in defect_keywords):
                                    start_search = max(0, idx - 3)
                                    preceding_context = words[start_search:idx]
                                    if any(neg in preceding_context for neg in negations):
                                        logger.debug(f"Negated defect complaint skipped: '{sentence}'")
                                        continue  
                                   
                                    has_defect_complaint = True
                                    logger.info(f"Active defect complaint detected in text: '{sentence}'")
                                    break
                            if has_defect_complaint:
                                break
                        if has_defect_complaint:
                            break


                    if not has_defect_complaint:
                        logger.info("M2 Defect: No defect keywords found in reviews. Overriding image classification to NORMAL.")
                        defect_result = {
                            "label": "normal",
                            "confidence": 1.0,
                            "uncertain": False,
                            "scores": {"normal": 1.0, "defective": 0.0},
                            "defect_type": "none",
                            "defect_type_confidence": 0.0,
                            "overlay_b64": None,
                        }
                    else:
                        logger.info("M2 Defect: Defect keyword found in reviews. Running ResNet classification on image...")
                        defect_result = self.defect.predict(
                            image_path,
                            generate_heatmap=generate_heatmap,
                        )

                    module_outputs["defect"] = defect_result
                    timing["defect"] = round(time.time() - t0, 2)
                    self._log_gpu_memory("post-defect")
                    logger.success(
                        f"M2 done in {timing['defect']}s | "
                        f"result: {defect_result['label']} "
                        f"({defect_result['confidence']:.1%})"
                    )
                except Exception as e:
                    logger.error(f"M2 Defect failed: {e}")
                    module_outputs["defect"] = None
            else:
                logger.info(
                    "M2 Defect: no image provided — "
                    "defect dimension will be excluded from quality score."
                )
                module_outputs["defect"] = None


        t0 = time.time()
        fusion_result = self.fusion.compute(
            sentiment_results   = module_outputs.get("sentiment"),
            defect_result       = module_outputs.get("defect"),
            fake_review_results = module_outputs.get("fake"),
            absa_results        = module_outputs.get("absa"),
            product_name        = product_name,
        )
        timing["fusion"] = round(time.time() - t0, 2)

        total_time = round(time.time() - start_time, 2)

        results.update({
            "quality_score":  fusion_result["quality_score"],
            "grade":          fusion_result["grade"],
            "breakdown":      fusion_result["breakdown"],
            "flags":          fusion_result["flags"],
            "summary":        fusion_result["summary"],
            "defect_source":  fusion_result.get("defect_source", "excluded"),
            "module_outputs": module_outputs,
            "timing":         {**timing, "total": total_time},
            "n_reviews":      len(reviews),
            "has_image":      image_path is not None,
            "device":         str(self.device),
        })

        if ood_detected:
            results["flags"].append("Out-of-distribution (OOD) reviews detected (gibberish/non-English)")

        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics({
                    "quality_score": results["quality_score"],
                    "fake_review_ratio": len([r for r in (module_outputs.get("fake") or []) if r.get("is_fake")]) / max(len(reviews), 1),
                    "sentiment_score": results["breakdown"].get("sentiment", 50.0),
                })
        except Exception as e:
            logger.debug(f"Could not log inference metrics to MLflow: {e}")

        logger.success(
            f"Analysis complete | "
            f"Score: {results['quality_score']}/100 | "
            f"Time: {total_time}s | "
            f"Reviews: {len(reviews)} | "
            f"Defect source: {results['defect_source']}"
        )
        return results

    def analyze_dataframe(
        self,
        df:          pd.DataFrame,
        text_col:    str = "clean_text",
        rating_col:  str = "rating",
        asin_col:    str = "asin",
        min_reviews: int = 5,
    ) -> pd.DataFrame:
    
        if asin_col not in df.columns:
            df = df.copy()
            df[asin_col] = "UNKNOWN"

        product_results = []
        asins = df[asin_col].unique()
        logger.info(f"Analyzing {len(asins)} products...")

        for i, asin in enumerate(asins, 1):
            product_df = df[df[asin_col] == asin]

            if len(product_df) < min_reviews:
                logger.debug(
                    f"Skipping {asin} — only {len(product_df)} reviews "
                    f"(min: {min_reviews})"
                )
                continue

            reviews = product_df[text_col].tolist()
            ratings = product_df[rating_col].tolist() \
                      if rating_col in product_df.columns else None

            logger.info(f"[{i}/{len(asins)}] {asin} — {len(reviews)} reviews")

            result = self.analyze(
                reviews=reviews,
                ratings=ratings,
                image_path=None,   
                asin=asin,
                product_name=f"Product {asin}",
                run_modules=["sentiment", "fake", "absa"],
                generate_heatmap=False,
            )

            product_results.append({
                "asin":               asin,
                "n_reviews":          result["n_reviews"],
                "quality_score":      result["quality_score"],
                "grade":              result["grade"],
                "sentiment_score":    result["breakdown"].get("sentiment", None),
                "authenticity_score": result["breakdown"].get("authenticity", None),
                "aspect_score":       result["breakdown"].get("aspect", None),
                "defect_score":       result["breakdown"].get("defect", None),
                "defect_source":      result["defect_source"],
                "n_flags":            len(result["flags"]),
                "flags":              "; ".join(result["flags"]),
                "summary":            result["summary"],
                "total_time_s":       result["timing"]["total"],
            })

        result_df = pd.DataFrame(product_results)
        if not result_df.empty:
            result_df = result_df.sort_values(
                "quality_score", ascending=False
            ).reset_index(drop=True)

        logger.success(
            f"Batch analysis complete | "
            f"{len(result_df)} products | "
            f"Avg score: {result_df['quality_score'].mean():.1f}"
            if not result_df.empty else "No products met minimum review threshold"
        )
        return result_df


if __name__ == "__main__":
    setup_logger()

    pipeline = AnalysisPipeline()

    test_reviews = [
        # Positive reviews
        "Battery life is great, lasts all day easily. Very impressed.",
        "Camera quality is stunning, takes incredible photos in low light.",
        "Build quality feels premium and sturdy. Worth every penny.",
        "Sound quality is amazing, bass is deep, highs are crisp.",
        "Setup was super easy, connected in 30 seconds. Love it.",
        "Price is very reasonable for the quality you get here.",
        "Comfortable to wear for long periods. No ear fatigue at all.",
        "Bluetooth range is excellent, works across my whole apartment.",
        "Very satisfied with this purchase. Would buy again without hesitation.",
        "Display is gorgeous, colours are vivid and sharp.",
        # Neutral reviews
        "It is okay, does what it says. Nothing special.",
        "Decent product for the price. Has some minor issues.",
        "Works fine. Battery could be better but acceptable.",
        "Average quality. Not amazing but not terrible either.",
        "Setup was a bit confusing but works okay after that.",
        # Negative reviews
        "Battery drains so fast, barely lasts 3 hours. Very disappointed.",
        "Bluetooth keeps disconnecting every 10 minutes. Very frustrating.",
        "Screen cracked after one week. Terrible build quality.",
        "Sound quality is terrible, no bass at all. Waste of money.",
        "Stopped working after two days. Complete waste of money.",
        "Build feels cheap and plasticky. Not worth the price.",
        "Connection drops constantly. Unusable for calls.",
        # Suspicious reviews (likely fake)
        "Great product! Highly recommend! Works as described! Five stars! Amazing!",
        "Perfect product! Amazing quality! Would recommend to everyone! Love it!",
        "Best product ever! Exceeded expectations! Very satisfied! Will buy again!",
        # More genuine mixed reviews
        "Good noise cancellation but the case feels flimsy.",
        "Battery lasts long but takes forever to charge.",
        "Sounds great for music but call quality is mediocre.",
        "Love the design but the touch controls are oversensitive.",
        "Great value. Not perfect but solid for the price point.",
    ]
    test_ratings = [
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 1,
        5, 5, 5,
        3, 3, 3, 3, 4,
    ]

    print("\n" + "="*60)
    print(f"Full Pipeline — {len(test_reviews)} reviews | No image")
    print("="*60)

    result = pipeline.analyze(
        reviews=test_reviews,
        ratings=test_ratings,
        image_path=None,    
        asin="B08XYZ123",
        product_name="Wireless Earbuds Pro",
        run_modules=["sentiment", "fake", "absa"],
        generate_heatmap=False,
    )

    print(f"\nQuality Score: {result['quality_score']}/100")
    print(f"Defect:        {result['defect_source']} (no image provided)")
    print(f"\nBreakdown:")
    for dim, score in result["breakdown"].items():
        bar = "█" * int(score / 5)
        print(f"  {dim:<15} {score:>5.1f}/100  {bar}")
    print(f"\nFlags:   {result['flags'] or ['none']}")
    print(f"Summary: {result['summary']}")