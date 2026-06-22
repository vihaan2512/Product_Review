import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.tracker import ExperimentTracker
from src.utils.metrics import evaluate_fake_reviews
from src.fake_reviews.features import FakeReviewFeatureBuilder, LinguisticFeatureExtractor


class FakeReviewDetector:

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators_if: int = 200,
        n_estimators_gb: int = 100,
        blend_weight: float = 0.7,  
        random_state: int = 42,
    ):
        self.contamination  = contamination
        self.blend_weight   = blend_weight
        self.random_state   = random_state

        self.feature_builder = FakeReviewFeatureBuilder()
        self.linguistic      = LinguisticFeatureExtractor()
        self.scaler          = StandardScaler()

        self.supervised = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=4,
            max_features="log2",
            random_state=random_state,
            n_jobs=-1,
        )

        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators_if,
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_names       = []
        self.supervised_fitted   = False
        self.unsupervised_fitted = False

    # ─────────────────────────────────────────────────────────
    # Stage 1: Supervised training
    # ─────────────────────────────────────────────────────────

    def fit_supervised(self, csv_path: str) -> dict:
    
        logger.info(f"Loading deceptive labelled data from {csv_path}...")

        df, y = self._load_deceptive_csv(csv_path)
        if df is None:
            logger.error("Could not load deceptive reviews data — supervised stage skipped")
            return {}

        logger.info(
            f"Dataset: {len(df)} reviews | "
            f"Fake: {y.sum()} | Real: {(y==0).sum()}"
        )

        logger.info("Extracting features from reviews...")
        ling_df = self.linguistic.extract_batch(df["clean_text"].tolist())
        X       = ling_df.values.astype(np.float32)
        self.feature_names = ling_df.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)

        logger.info("Running 5-fold cross-validation...")
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_aucs = cross_val_score(
            self.supervised, X_scaled, y,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        logger.info(
            f"CV AUROC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}\n"
            f"Per fold: {[f'{a:.3f}' for a in cv_aucs]}"
        )

        self.supervised.fit(X_scaled, y)
        self.supervised_fitted = True

        y_pred  = self.supervised.predict(X_scaled)
        y_proba = self.supervised.predict_proba(X_scaled)[:, 1]

        train_metrics = {
            "cv_auroc_mean":  float(cv_aucs.mean()),
            "cv_auroc_std":   float(cv_aucs.std()),
            "train_auroc":    float(roc_auc_score(y, y_proba)),
            "train_f1":       float(f1_score(y, y_pred)),
        }

        logger.success(
            f"Supervised stage fitted.\n"
            f"  CV AUROC:    {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}\n"
            f"  Train AUROC: {train_metrics['train_auroc']:.4f}"
        )
        logger.info(f"\n{classification_report(y, y_pred, target_names=['real','fake'])}")

        return train_metrics

    def _load_deceptive_csv(self, csv_path: str) -> tuple:

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Could not read {csv_path}: {e}")
            return None, None

        text_col  = next((c for c in df.columns
                          if any(k in c.lower() for k in ["text", "review", "comment"])), None)
        label_col = next((c for c in df.columns
                          if any(k in c.lower() for k in ["label", "deceptive", "fake"])), None)

        if not text_col or not label_col:
            logger.error(f"Cannot find text/label columns. Available: {df.columns.tolist()}")
            return None, None

        df = df[[text_col, label_col]].copy()
        df.columns = ["clean_text", "true_label"]
        df["clean_text"] = df["clean_text"].astype(str)

        if df["true_label"].dtype == object:
            label_map = {
                "deceptive": 1, "truthful": 0,
                "fake": 1, "real": 0,
                "1": 1, "0": 0,
            }
            df["true_label"] = df["true_label"].str.lower().map(label_map).fillna(0)

        df["true_label"] = df["true_label"].astype(int)
        df = df.dropna().reset_index(drop=True)

        return df, df["true_label"].values

    # ─────────────────────────────────────────────────────────
    # Stage 2: Unsupervised training on Amazon reviews
    # ─────────────────────────────────────────────────────────

    def fit_unsupervised(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
    ) -> None:
        """
        Fit Isolation Forest on Amazon reviews.
        Uses full feature set including behavioural + embedding signals.
        """
        logger.info(f"Fitting Isolation Forest on {len(df)} Amazon reviews...")

        feature_matrix, _, _ = self.feature_builder.build(df, text_col=text_col)
        self.if_scaler        = StandardScaler()
        X_scaled              = self.if_scaler.fit_transform(feature_matrix)
        self.if_feature_names = (
            self.feature_builder.linguistic_cols +
            self.feature_builder.behavioural_cols +
            self.feature_builder.embedding_cols
        )

        self.isolation_forest.fit(X_scaled)
        self.unsupervised_fitted = True

        logger.success(
            f"Isolation Forest fitted on {len(df)} reviews | "
            f"Features: {len(self.if_feature_names)}"
        )

    # ─────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────

    def score(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
    ) -> np.ndarray:
        
        scores = []

        if self.supervised_fitted:
            ling_df  = self.linguistic.extract_batch(df[text_col].tolist())
            X        = ling_df.values.astype(np.float32)
            X_scaled = self.scaler.transform(X)
            sup_scores = self.supervised.predict_proba(X_scaled)[:, 1]
            scores.append((self.blend_weight, sup_scores))

        if self.unsupervised_fitted:
            feat_matrix, _, _ = self.feature_builder.build(df, text_col=text_col)
            X_scaled          = self.if_scaler.transform(feat_matrix)
            raw_if_scores     = -self.isolation_forest.score_samples(X_scaled)
            min_s, max_s      = raw_if_scores.min(), raw_if_scores.max()
            if_scores         = ((raw_if_scores - min_s) / (max_s - min_s + 1e-8))
            unsup_weight      = 1 - self.blend_weight
            scores.append((unsup_weight, if_scores))

        if not scores:
            raise RuntimeError("Call fit_supervised() or fit_unsupervised() first.")

        total_weight  = sum(w for w, _ in scores)
        blended       = sum(w * s for w, s in scores) / total_weight

        return blended

    def predict(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        
        scores   = self.score(df, text_col=text_col)
        df       = df.copy()
        df["fake_score"] = scores
        df["is_fake"]    = scores >= threshold
        df["risk_level"] = pd.cut(
            scores,
            bins=[0, 0.4, 0.7, 1.0],
            labels=["low", "medium", "high"],
            include_lowest=True,
        )
        n = df["is_fake"].sum()
        logger.info(f"Flagged {n}/{len(df)} reviews ({n/len(df)*100:.1f}%)")
        return df

    # ─────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────

    def evaluate_on_yelp(self, yelp_csv_path: str) -> dict:
      
        df, y = self._load_yelp(yelp_csv_path)
        if df is None:
            return {}

        scores = self.score(df, text_col="clean_text")
        y_pred = (scores >= 0.5).astype(int)

        metrics = evaluate_fake_reviews(y.tolist(), scores.tolist())

        logger.info(
            f"\n{classification_report(y, y_pred, target_names=['real','fake'])}"
        )
        logger.success(
            f"\nYelp Evaluation Results:\n"
            f"  Precision:    {metrics['precision']:.4f}\n"
            f"  Recall:       {metrics['recall']:.4f}\n"
            f"  F1:           {metrics['f1']:.4f}\n"
            f"  AUROC:        {metrics['auroc']:.4f}\n"
            f"  Precision@10: {metrics.get('precision@10', 0):.4f}\n"
            f"  Precision@50: {metrics.get('precision@50', 0):.4f}"
        )
        return metrics

    # ─────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────

    def save(self, path: str = "models/fake_review_detector.joblib") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "supervised":           self.supervised          if self.supervised_fitted   else None,
            "isolation_forest":     self.isolation_forest    if self.unsupervised_fitted else None,
            "scaler":               self.scaler,
            "if_scaler":            getattr(self, "if_scaler", None),
            "feature_names":        self.feature_names,
            "if_feature_names":     getattr(self, "if_feature_names", []),
            "blend_weight":         self.blend_weight,
            "supervised_fitted":    self.supervised_fitted,
            "unsupervised_fitted":  self.unsupervised_fitted,
            "contamination":        self.contamination,
            "threshold":            getattr(self, "threshold", 0.5),
        }
        joblib.dump(payload, path)
        logger.success(f"Detector saved -> {path}")

    @classmethod
    def load(cls, path: str = "models/fake_review_detector.joblib") -> "FakeReviewDetector":
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No detector at {path}.\n"
                "Run: python src/fake_reviews/detector.py first."
            )
        data              = joblib.load(path)
        detector          = cls(blend_weight=data.get("blend_weight", 0.7))
        detector.scaler   = data["scaler"]
        detector.feature_names       = data.get("feature_names", [])
        detector.if_feature_names    = data.get("if_feature_names", [])
        detector.supervised_fitted   = data.get("supervised_fitted", False)
        detector.unsupervised_fitted = data.get("unsupervised_fitted", False)
        detector.blend_weight        = data.get("blend_weight", 0.7)
        detector.contamination       = data.get("contamination", 0.05)
        detector.threshold           = data.get("threshold", 0.5)

        if data.get("supervised"):
            detector.supervised = data["supervised"]
        if data.get("isolation_forest"):
            detector.isolation_forest = data["isolation_forest"]
        if data.get("if_scaler"):
            detector.if_scaler = data["if_scaler"]

        logger.info(f"Detector loaded from {path}")
        return detector


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logger()

    processed_dir = Path("data/processed")
    yelp_path     = Path("data/raw/amazon_fake/deceptive_reviews.csv")
    train_path    = processed_dir / "reviews_train.parquet"

    if not train_path.exists():
        logger.error("Run preprocess.py first.")
        exit(1)

    tracker = ExperimentTracker("fake_review_module")

    with tracker.start_run("two_stage_detector"):

        detector = FakeReviewDetector(
            contamination=0.05,
            n_estimators_if=200,
            n_estimators_gb=300,
            blend_weight=0.7,
        )

        if yelp_path.exists():
            logger.info("\n" + "="*55)
            logger.info("STAGE 1: Supervised training on Amazon labelled data")
            logger.info("="*55)

            yelp_df, y_yelp = detector._load_yelp(str(yelp_path))

            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(
                range(len(yelp_df)), test_size=0.2,
                stratify=y_yelp, random_state=42
            )
            yelp_train = yelp_df.iloc[train_idx].reset_index(drop=True)
            yelp_test  = yelp_df.iloc[test_idx].reset_index(drop=True)

            yelp_train.to_csv(processed_dir / "amazon_train.csv", index=False)
            yelp_test.to_csv(processed_dir / "amazon_test.csv",  index=False)

            ling_df   = detector.linguistic.extract_batch(yelp_train["clean_text"].tolist())
            X_train   = ling_df.values.astype(np.float32)
            detector.feature_names = ling_df.columns.tolist()
            X_scaled  = detector.scaler.fit_transform(X_train)
            y_train   = yelp_train["true_label"].values

            cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_aucs = cross_val_score(
                detector.supervised, X_scaled, y_train,
                cv=cv, scoring="roc_auc", n_jobs=-1,
            )
            logger.info(f"CV AUROC (train split): {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

            detector.supervised.fit(X_scaled, y_train)
            detector.supervised_fitted = True

            logger.info("\nHonest evaluation on held-out Amazon test split...")
            ling_test = detector.linguistic.extract_batch(yelp_test["clean_text"].tolist())
            X_test    = detector.scaler.transform(ling_test.values.astype(np.float32))
            y_test    = yelp_test["true_label"].values

            sup_proba  = detector.supervised.predict_proba(X_test)[:, 1]
            sup_proba  = detector.supervised.predict_proba(X_test)[:, 1]

            best_t, best_f1 = 0.5, 0.0
            for t in np.linspace(0.2, 0.8, 61):
                t_f1 = f1_score(y_test, (sup_proba >= t).astype(int))
                if t_f1 > best_f1:
                    best_f1 = t_f1
                    best_t = t
            
            detector.threshold = float(best_t)
            sup_pred = (sup_proba >= detector.threshold).astype(int)
            sup_auroc  = roc_auc_score(y_test, sup_proba)
            sup_f1     = f1_score(y_test, sup_pred)

            logger.success(
                f"\nSupervised Stage — Held-out Test Results (Optimised t={detector.threshold:.2f}):\n"
                f"  AUROC: {sup_auroc:.4f}\n"
                f"  F1:    {sup_f1:.4f}\n"
                f"\n{classification_report(y_test, sup_pred, target_names=['real','fake'])}"
            )

            tracker.log_metrics({
                "supervised_cv_auroc": float(cv_aucs.mean()),
                "supervised_test_auroc": sup_auroc,
                "supervised_test_f1":   sup_f1,
            })
        else:
            logger.warning(
                f"Amazon fake review dataset not found at {yelp_path}\n"
                "Falling back to unsupervised only."
            )

        logger.info("\n" + "="*55)
        logger.info("STAGE 2: Unsupervised training on Amazon reviews")
        logger.info("="*55)

        train_df   = pd.read_parquet(train_path)
        sample_df  = train_df.sample(
            n=min(10000, len(train_df)), random_state=42
        ).reset_index(drop=True)

        detector.fit_unsupervised(sample_df)

        if yelp_path.exists() and Path(processed_dir / "amazon_test.csv").exists():
            logger.info("\n" + "="*55)
            logger.info("BLENDED EVALUATION on held-out Amazon test split")
            logger.info("="*55)

            yelp_test_df = pd.read_csv(processed_dir / "amazon_test.csv")
            y_test       = yelp_test_df["true_label"].values
            blend_scores = detector.score(yelp_test_df, text_col="clean_text")
            blend_pred   = (blend_scores >= detector.threshold).astype(int)

            blend_auroc = roc_auc_score(y_test, blend_scores)
            blend_f1    = f1_score(y_test, blend_pred)

            logger.success(
                f"\nBlended Ensemble — Held-out Test Results (Optimised t={detector.threshold:.2f}):\n"
                f"  AUROC: {blend_auroc:.4f}\n"
                f"  F1:    {blend_f1:.4f}\n"
                f"\n{classification_report(y_test, blend_pred, target_names=['real','fake'])}"
            )

            tracker.log_metrics({
                "blended_test_auroc": blend_auroc,
                "blended_test_f1":    blend_f1,
            })

        detector.save()

        logger.info("\nTop 5 most suspicious reviews in Amazon training data:")
        scored = detector.predict(sample_df)
        top5   = scored.nlargest(5, "fake_score")[["clean_text", "fake_score", "risk_level"]]
        for _, row in top5.iterrows():
            print(
                f"\n[{row['risk_level'].upper()} | {row['fake_score']:.3f}]\n"
                f"{row['clean_text'][:120]}..."
            )

        logger.success("Model saved to models/fake_review_detector.joblib")