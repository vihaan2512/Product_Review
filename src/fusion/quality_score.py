from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.logger import setup_logger


DEFAULT_WEIGHTS = {
    "sentiment":    0.35,
    "defect":       0.25,
    "authenticity": 0.20,
    "aspect":       0.20,
}

GRADE_THRESHOLDS = [
    (93, "A+"), (90, "A"), (87, "A-"),
    (83, "B+"), (80, "B"), (77, "B-"),
    (73, "C+"), (70, "C"), (67, "C-"),
    (60, "D"),  (0,  "F"),
]

DEFECT_KEYWORDS = {
    "critical": [
        "broken", "cracked", "shattered", "defective", "damaged",
        "stopped working", "dead on arrival", "doa", "fell apart",
        "broke after", "stopped after", "not working", "doesnt work",
        "doesn't work", "malfunctioning",
    ],
    "moderate": [
        "scratched", "dent", "dented", "bent", "loose", "wobbly",
        "peeling", "fading", "discolored", "misaligned", "warped",
        "chipped", "worn out",
    ],
    "minor": [
        "small scratch", "minor defect", "tiny crack",
        "slight damage", "cosmetic",
    ],
}

DEFECT_RELATED_ASPECTS = [
    "build quality", "durability", "construction", "material",
    "finish", "hardware", "physical", "body", "casing", "quality",
    "build", "craftsmanship",
]


class QualityScoreFusion:

    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._validate_weights()
        self._train_learned_fusion_models()
        logger.info("QualityScoreFusion initialised with Learned Fusion Models.")

    def _validate_weights(self) -> None:
        required = {"sentiment", "defect", "authenticity", "aspect"}
        missing  = required - set(self.weights.keys())
        if missing:
            raise ValueError(f"Missing weight keys: {missing}")

    def _train_learned_fusion_models(self):
        from sklearn.linear_model import Ridge
        import numpy as np
        import sqlite3
        import json
        from pathlib import Path

        db_path = Path("data/reports.db")
        real_rows = []
        
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path), timeout=5.0)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT breakdown, actual_rating FROM product_reports WHERE actual_rating IS NOT NULL")
                real_rows = [dict(row) for row in cur.fetchall()]
                conn.close()
            except Exception as e:
                logger.warning(f"Could not load real ratings for learned fusion: {e}")

        X_a_real, y_a_real = [], []
        X_b_real, y_b_real = [], []

        for row in real_rows:
            try:
                bd = json.loads(row["breakdown"])
                actual = float(row["actual_rating"])
                
                s = bd.get("sentiment")
                f = bd.get("authenticity")
                a = bd.get("aspect")
                d = bd.get("defect")
                
                if s is not None and f is not None and a is not None:
                    X_b_real.append([s, f, a])
                    y_b_real.append(actual)
                    
                    if d is not None and d != 0.0:
                        X_a_real.append([s, d, f, a])
                        y_a_real.append(actual)
            except Exception:
                continue

        if len(X_a_real) >= 50:
            logger.info(f"Training Learned Fusion Model A on {len(X_a_real)} REAL database records...")
            self.model_full = Ridge(alpha=1.0)
            self.model_full.fit(np.array(X_a_real), np.array(y_a_real))
        else:
            logger.info("Insufficient Model A database records (< 50). Training on synthetic ratings fallback...")
            np.random.seed(42)
            X_a = np.random.uniform(30.0, 100.0, size=(1500, 4))
            y_a = (X_a[:, 0] * 0.35 + X_a[:, 1] * 0.25 + X_a[:, 2] * 0.20 + X_a[:, 3] * 0.20)
            y_a = y_a - (100.0 - X_a[:, 2]) * 0.05 - (100.0 - X_a[:, 1]) * 0.05
            y_a = np.clip(y_a, 0.0, 100.0)
            self.model_full = Ridge(alpha=1.0)
            self.model_full.fit(X_a, y_a)

        if len(X_b_real) >= 50:
            logger.info(f"Training Learned Fusion Model B on {len(X_b_real)} REAL database records...")
            self.model_no_defect = Ridge(alpha=1.0)
            self.model_no_defect.fit(np.array(X_b_real), np.array(y_b_real))
        else:
            logger.info("Insufficient Model B database records (< 50). Training on synthetic ratings fallback...")
            X_b = np.random.uniform(30.0, 100.0, size=(1500, 3))
            y_b = (X_b[:, 0] * (0.35/0.75) + X_b[:, 1] * (0.20/0.75) + X_b[:, 2] * (0.20/0.75))
            y_b = y_b - (100.0 - X_b[:, 1]) * 0.08
            y_b = np.clip(y_b, 0.0, 100.0)
            self.model_no_defect = Ridge(alpha=1.0)
            self.model_no_defect.fit(X_b, y_b)

        logger.info(f"Learned Fusion Model A Coefficients: {self.model_full.coef_} (Intercept: {self.model_full.intercept_:.2f})")
        logger.info(f"Learned Fusion Model B Coefficients: {self.model_no_defect.coef_} (Intercept: {self.model_no_defect.intercept_:.2f})")


    def sentiment_subscore(self, sentiment_results: list[dict]) -> tuple:
        if not sentiment_results:
            return 50.0, {"reason": "no reviews", "n_reviews": 0}

        label_to_score = {"positive": 100.0, "neutral": 50.0, "negative": 0.0}
        scores, weights = [], []

        for r in sentiment_results:
            label      = r.get("label", "neutral")
            confidence = r.get("confidence", 0.5)
            uncertain  = r.get("uncertain", False)
            base_score = label_to_score.get(label, 50.0)
            weight     = confidence * (0.5 if uncertain else 1.0)
            scores.append(base_score)
            weights.append(weight)

        if sum(weights) == 0:
            return 50.0, {"reason": "all uncertain"}

        weighted_avg = float(np.average(scores, weights=weights))

        label_counts = {}
        for r in sentiment_results:
            l = r.get("label", "neutral")
            label_counts[l] = label_counts.get(l, 0) + 1

        n = len(sentiment_results)
        metadata = {
            "n_reviews":    n,
            "label_counts": label_counts,
            "avg_score":    round(weighted_avg, 2),
            "pct_positive": round(label_counts.get("positive", 0) / n * 100, 1),
            "pct_negative": round(label_counts.get("negative", 0) / n * 100, 1),
            "pct_neutral":  round(label_counts.get("neutral",  0) / n * 100, 1),
        }
        return round(weighted_avg, 2), metadata

    def defect_subscore_from_image(self, defect_result: dict) -> tuple:
        defect_prob = defect_result.get("scores", {}).get("defective", 0.5)
        confidence  = defect_result.get("confidence", 0.5)
        uncertain   = defect_result.get("uncertain", False)

        base_score = (1.0 - defect_prob) * 100
        if uncertain:
            base_score = 0.5 * base_score + 0.5 * 75.0

        metadata = {
            "source":      "image_analysis",
            "label":       defect_result.get("label", "unknown"),
            "defect_prob": round(defect_prob, 4),
            "confidence":  round(confidence, 4),
            "uncertain":   uncertain,
            "defect_type": defect_result.get("defect_type", None),
        }
        return round(float(base_score), 2), metadata

    def defect_subscore_from_text(self, reviews: list[str]) -> tuple:
        severity_weights = {"critical": 1.0, "moderate": 0.5, "minor": 0.2}
        total_penalty    = 0.0
        defect_mentions  = []

        for review in reviews:
            text_lower = review.lower()
            for severity, keywords in DEFECT_KEYWORDS.items():
                for kw in keywords:
                    if kw in text_lower:
                        total_penalty += severity_weights[severity]
                        defect_mentions.append({
                            "keyword":  kw,
                            "severity": severity,
                        })
                        break  

        penalty_score = min(total_penalty / max(len(reviews), 1) * 50, 100)
        sub_score     = max(0.0, 100.0 - penalty_score)

        metadata = {
            "source":          "text_mining",
            "defect_mentions": defect_mentions[:5],
            "n_mentions":      len(defect_mentions),
            "n_reviews":       len(reviews),
            "penalty_score":   round(penalty_score, 2),
        }
        return round(sub_score, 2), metadata

    def defect_subscore_from_absa(self, absa_results: dict) -> tuple:
        relevant = {
            k: v for k, v in absa_results.items()
            if any(d in k.lower() for d in DEFECT_RELATED_ASPECTS)
        }
        if not relevant:
            return None, {"source": "absa_proxy", "reason": "no defect aspects found"}

        scores    = [v.get("mean_score", 0) for v in relevant.values()]
        avg_score = float(np.mean(scores))
        sub_score = (avg_score + 1.0) / 2.0 * 100.0

        return round(float(np.clip(sub_score, 0, 100)), 2), {
            "source":           "absa_proxy",
            "relevant_aspects": list(relevant.keys()),
            "avg_aspect_score": round(avg_score, 4),
        }

    def defect_subscore(
        self,
        defect_result: dict       = None,
        reviews:       list[str]  = None,
        absa_results:  dict       = None,
    ) -> tuple:
    
        if defect_result is not None:
            image_score, image_meta = self.defect_subscore_from_image(defect_result)
            
            text_score = 100.0
            if reviews and len(reviews) > 0:
                text_score, _ = self.defect_subscore_from_text(reviews)
                
            absa_score = 100.0
            if absa_results:
                absa_score, _ = self.defect_subscore_from_absa(absa_results)
                if absa_score is None:
                    absa_score = 100.0
                    
            min_text_absa = min(text_score, absa_score)
            if image_score > 75.0 and min_text_absa < 70.0:
                penalty = (image_score - min_text_absa) * 0.4
                new_score = max(0.0, image_score - penalty)
                image_meta["cross_verified_penalty"] = round(penalty, 2)
                image_meta["contradiction_detected"] = True
                image_meta["reason"] = f"Image shows normal but text/aspects indicate defect (penalty: -{penalty:.1f})"
                return round(new_score, 2), image_meta
                
            return image_score, image_meta

        return None, {"source": "none", "reason": "Excluded from scoring — no image analyzed"}

    def authenticity_subscore(self, fake_review_results: list[dict]) -> tuple:
        if not fake_review_results:
            return 70.0, {"reason": "no fake detection run"}

        fake_scores  = [r.get("fake_score", 0.5) for r in fake_review_results]
        n_flagged    = sum(1 for r in fake_review_results if r.get("is_fake", False))
        n_total      = len(fake_review_results)
        avg_fake     = float(np.mean(fake_scores))
        auth_raw     = (1.0 - avg_fake) * 100
        fake_prop    = n_flagged / n_total
        penalty      = fake_prop * 30
        sub_score    = max(0.0, auth_raw - penalty)

        metadata = {
            "n_reviews":       n_total,
            "n_flagged":       n_flagged,
            "pct_flagged":     round(fake_prop * 100, 1),
            "avg_fake_score":  round(avg_fake, 4),
            "high_risk_count": sum(1 for r in fake_review_results
                                   if r.get("risk_level") == "high"),
        }
        return round(float(sub_score), 2), metadata

    def aspect_subscore(self, absa_results: dict) -> tuple:
        if not absa_results:
            return 65.0, {"reason": "no ABSA results", "n_aspects": 0}

        mean_scores = [
            v.get("mean_score", v.get("score", 0.0))
            for v in absa_results.values()
        ]
        if not mean_scores:
            return 65.0, {"reason": "empty ABSA results"}

        avg_score = float(np.mean(mean_scores))
        sub_score = float(np.clip((avg_score + 1.0) / 2.0 * 100.0, 0, 100))

        sorted_aspects = sorted(
            absa_results.items(),
            key=lambda x: x[1].get("mean_score", x[1].get("score", 0)),
        )
        metadata = {
            "n_aspects":     len(absa_results),
            "avg_score":     round(avg_score, 4),
            "best_aspects":  [a for a, _ in sorted_aspects[-3:]],
            "worst_aspects": [a for a, _ in sorted_aspects[:3]],
        }
        return round(sub_score, 2), metadata

    def compute(
        self,
        sentiment_results:    list[dict] = None,
        defect_result:        dict       = None,
        fake_review_results:  list[dict] = None,
        absa_results:         dict       = None,
        product_name:         str        = "Product",
    ) -> dict:
       
        sub_scores = {}
        metadata   = {}
        flags      = []

        review_texts = []
        if sentiment_results:
            review_texts = [r.get("text", "") for r in sentiment_results
                            if r.get("text")]

        if sentiment_results is not None:
            score, meta = self.sentiment_subscore(sentiment_results)
            sub_scores["sentiment"] = score
            metadata["sentiment"]   = meta
            if meta.get("pct_negative", 0) > 40:
                flags.append(f"{meta['pct_negative']:.0f}% negative reviews")

        defect_score, defect_meta = self.defect_subscore(
            defect_result=defect_result,
            reviews=review_texts or None,
            absa_results=absa_results,
        )
        metadata["defect"] = defect_meta

        if defect_score is not None:
            sub_scores["defect"] = defect_score
            if defect_meta.get("contradiction_detected"):
                flags.append(
                    "contradiction: image shows normal but text/aspects indicate physical defect"
                )
            if defect_meta.get("source") == "image_analysis":
                if defect_meta.get("label") == "defective" and \
                   not defect_meta.get("uncertain"):
                    flags.append(
                        f"defect detected: {defect_meta.get('defect_type', 'unknown')}"
                    )
            elif defect_meta.get("n_mentions", 0) > 0:
                flags.append(
                    f"{defect_meta['n_mentions']} defect mentions in reviews"
                )

        if fake_review_results is not None:
            score, meta = self.authenticity_subscore(fake_review_results)
            sub_scores["authenticity"] = score
            metadata["authenticity"]   = meta
            if meta.get("pct_flagged", 0) > 15:
                flags.append(f"{meta['pct_flagged']:.0f}% reviews flagged as suspicious")
            if meta.get("high_risk_count", 0) > 0:
                flags.append(f"{meta['high_risk_count']} high-risk reviews detected")

        if absa_results is not None:
            score, meta = self.aspect_subscore(absa_results)
            sub_scores["aspect"] = score
            metadata["aspect"]   = meta
            if meta.get("worst_aspects"):
                flags.append(f"weak aspects: {', '.join(meta['worst_aspects'][:2])}")

        defaults = {
            "sentiment":    65.0,
            "authenticity": 70.0,
            "aspect":       65.0,
        }
        for key, default in defaults.items():
            if key not in sub_scores:
                sub_scores[key] = default

        if "defect" not in sub_scores:
            inp = np.array([[sub_scores["sentiment"], sub_scores["authenticity"], sub_scores["aspect"]]])
            quality_score = float(self.model_no_defect.predict(inp)[0])
            
            coef = self.model_no_defect.coef_
            effective_weights = {
                "sentiment":    float(coef[0]),
                "authenticity": float(coef[1]),
                "aspect":       float(coef[2]),
            }
        else:
            inp = np.array([[sub_scores["sentiment"], sub_scores["defect"], sub_scores["authenticity"], sub_scores["aspect"]]])
            quality_score = float(self.model_full.predict(inp)[0])
            
            coef = self.model_full.coef_
            effective_weights = {
                "sentiment":    float(coef[0]),
                "defect":       float(coef[1]),
                "authenticity": float(coef[2]),
                "aspect":       float(coef[3]),
            }

        quality_score = float(np.clip(quality_score, 0.0, 100.0))

        grade = ""

        summary = self._generate_summary(
            quality_score, grade, sub_scores, metadata, product_name,
            defect_available=defect_score is not None,
            defect_source=defect_meta.get("source", "none"),
        )

        logger.info(
            f"Quality score: {quality_score:.1f}/100 | "
            f"Defect source: {defect_meta.get('source', 'excluded')} | "
            f"Flags: {len(flags)}"
        )

        return {
            "quality_score":   round(quality_score, 1),
            "grade":           grade,
            "breakdown":       {k: round(v, 1) for k, v in sub_scores.items()},
            "effective_weights": effective_weights,
            "flags":           flags,
            "summary":         summary,
            "metadata":        metadata,
            "defect_source":   defect_meta.get("source", "excluded"),
        }

    def _generate_summary(
        self,
        score: float,
        grade: str,
        sub_scores: dict,
        metadata: dict,
        product_name: str,
        defect_available: bool = True,
        defect_source: str = "none",
    ) -> str:
        lines = []
        if score >= 80:
            lines.append(
                f"{product_name} scores {score:.0f}/100 — strong overall quality."
            )
        elif score >= 65:
            lines.append(
                f"{product_name} scores {score:.0f}/100 — decent quality with some concerns."
            )
        else:
            lines.append(
                f"{product_name} scores {score:.0f}/100 — significant quality issues detected."
            )

        sent_meta = metadata.get("sentiment", {})
        if sent_meta.get("pct_positive"):
            lines.append(
                f"{sent_meta['pct_positive']:.0f}% of {sent_meta.get('n_reviews', 0)} reviews are positive."
            )

        if not defect_available:
            source_msg = {
                "text_mining": "Defect score estimated from review text (no image provided).",
                "absa_proxy":  "Defect score estimated from build quality aspects (no image provided).",
                "none":        "No product image provided — defect dimension excluded from score.",
            }
            lines.append(source_msg.get(defect_source, ""))

        if sub_scores:
            best_dim  = max(sub_scores, key=sub_scores.get)
            worst_dim = min(sub_scores, key=sub_scores.get)
            if sub_scores[best_dim] - sub_scores[worst_dim] > 15:
                dim_labels = {
                    "sentiment":    "customer sentiment",
                    "defect":       "product condition",
                    "authenticity": "review authenticity",
                    "aspect":       "aspect-level quality",
                }
                lines.append(
                    f"Strongest: {dim_labels.get(best_dim, best_dim)} "
                    f"({sub_scores[best_dim]:.0f}/100). "
                    f"Weakest: {dim_labels.get(worst_dim, worst_dim)} "
                    f"({sub_scores[worst_dim]:.0f}/100)."
                )

        return " ".join(l for l in lines if l)

    def update_weights(self, new_weights: dict) -> None:
        self.weights.update(new_weights)
        self._validate_weights()
        logger.info(f"Weights updated: {self.weights}")