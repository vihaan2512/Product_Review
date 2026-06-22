import re
import string
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


GENERIC_PHRASES = [
    "highly recommend", "great product", "works as described",
    "exactly as advertised", "fast shipping", "good quality",
    "worth the money", "five stars", "love this product",
    "would recommend", "very happy", "exceeded expectations",
    "perfect product", "amazing product", "works great",
    "does what it says", "great value", "very satisfied",
    "definitely recommend", "best product ever",
]


# ─────────────────────────────────────────────────────────────
# Signal 1: Linguistic Features
# ─────────────────────────────────────────────────────────────

class LinguisticFeatureExtractor:
   
    def __init__(self):
        self.generic_phrases = GENERIC_PHRASES

    def extract(self, text: str) -> dict:

        if not isinstance(text, str) or len(text.strip()) == 0:
            return self._empty_features()

        text_lower = text.lower()
        words      = text_lower.split()
        sentences  = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        n_words    = len(words)
        n_chars    = len(text)

        if n_words == 0:
            return self._empty_features()

        generic_count = sum(1 for p in self.generic_phrases if p in text_lower)

        type_token_ratio = len(set(words)) / n_words

        punct_count = sum(1 for c in text if c in string.punctuation)
        punct_ratio = punct_count / n_chars if n_chars > 0 else 0

        excl_ratio = text.count("!") / len(sentences) if sentences else 0

        caps_ratio = sum(1 for c in text if c.isupper()) / n_chars if n_chars > 0 else 0

        avg_word_len  = float(np.mean([len(w) for w in words])) if words else 0
        sent_lengths  = [len(s.split()) for s in sentences if s.split()]
        avg_sent_len  = float(np.mean(sent_lengths)) if sent_lengths else 0

        starts_with_i  = int(text_lower.strip().startswith("i "))
        has_price      = int(any(w in text_lower for w in
                                 ["price", "cost", "cheap", "expensive", "worth"]))
        has_comparison = int(any(w in text_lower for w in
                                 ["better", "best", "worse", "worst", "compared"]))
        has_extreme    = int(any(w in text_lower for w in
                                 ["amazing", "terrible", "perfect", "horrible",
                                  "outstanding", "awful", "excellent", "dreadful"]))

        return {
            "generic_phrase_count": generic_count,
            "generic_phrase_ratio": generic_count / n_words,
            "exclamation_ratio":    excl_ratio,
            "caps_ratio":           caps_ratio,
            "avg_word_length":      avg_word_len,
            "type_token_ratio":     type_token_ratio,
            "punctuation_ratio":    punct_ratio,
            "sentence_count":       len(sentences),
            "avg_sentence_length":  avg_sent_len,
            "word_count":           n_words,
            "starts_with_i":        starts_with_i,
            "has_price_mention":    has_price,
            "has_comparison":       has_comparison,
            "extreme_sentiment":    has_extreme,
        }

    def _empty_features(self) -> dict:
        return {k: 0.0 for k in [
            "generic_phrase_count", "generic_phrase_ratio",
            "exclamation_ratio", "caps_ratio", "avg_word_length",
            "type_token_ratio", "punctuation_ratio", "sentence_count",
            "avg_sentence_length", "word_count", "starts_with_i",
            "has_price_mention", "has_comparison", "extreme_sentiment",
        ]}

    def extract_batch(self, texts: list) -> pd.DataFrame:
        return pd.DataFrame([self.extract(t) for t in texts])


# ─────────────────────────────────────────────────────────────
# Signal 2: Behavioural Features
# ─────────────────────────────────────────────────────────────

class BehaviouralFeatureExtractor:

    def extract(self, df: pd.DataFrame) -> tuple:
        df = df.copy()

        if "verified_purchase" in df.columns:
            df["is_verified"] = df["verified_purchase"].astype(int)
        else:
            df["is_verified"] = 0

        if "asin" in df.columns and "rating" in df.columns:
            product_mean       = df.groupby("asin")["rating"].transform("mean")
            df["rating_deviation"] = (df["rating"] - product_mean).abs()
        else:
            df["rating_deviation"] = 0.0

        if "reviewerID" in df.columns:
            counts = df["reviewerID"].map(df["reviewerID"].value_counts())
            df["reviewer_review_count_log"] = np.log1p(counts.fillna(1))
        else:
            df["reviewer_review_count_log"] = 0.0

        if "asin" in df.columns and "unixReviewTime" in df.columns:
            df["burst_score"] = df.groupby("asin")["unixReviewTime"].transform(
                self._compute_burst_score
            )
        else:
            df["burst_score"] = 0.0

        if "rating" in df.columns:
            df["is_extreme_rating"] = df["rating"].apply(
                lambda r: 1 if r in [1, 5] else 0
            )
        else:
            df["is_extreme_rating"] = 0

        if "asin" in df.columns:
            pcounts = df["asin"].map(df["asin"].value_counts())
            df["product_review_count_log"] = np.log1p(pcounts.fillna(1))
        else:
            df["product_review_count_log"] = 0.0

        behavioural_cols = [
            "is_verified", "rating_deviation",
            "reviewer_review_count_log", "burst_score",
            "is_extreme_rating", "product_review_count_log",
        ]
        logger.info(f"Behavioural features extracted for {len(df)} reviews")
        return df, behavioural_cols

    def _compute_burst_score(self, timestamps: pd.Series) -> pd.Series:
    
        scores = pd.Series(index=timestamps.index, dtype=float)
        for idx, ts in timestamps.items():
            burst_count  = ((timestamps >= ts - 86400) & (timestamps <= ts + 86400)).sum()
            scores[idx]  = float(burst_count)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores


# ─────────────────────────────────────────────────────────────
# Signal 3: Embedding Features
# ─────────────────────────────────────────────────────────────

class EmbeddingFeatureExtractor:
   
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Sentence transformer loaded: {model_name}")
        except ImportError:
            raise ImportError(
                "Run: pip install sentence-transformers"
            )

    def encode(self, texts: list, batch_size: int = 64) -> np.ndarray:

        logger.info(f"Encoding {len(texts)} reviews...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.success(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    def compute_similarity_features(
        self, df: pd.DataFrame, embeddings: np.ndarray,
        asin_col: str = "asin",
    ) -> tuple:

        df        = df.copy().reset_index(drop=True)
        mean_sims = np.zeros(len(df))
        max_sims  = np.zeros(len(df))

        if asin_col not in df.columns:
            df["mean_cosine_similarity"] = 0.0
            df["max_cosine_similarity"]  = 0.0
            return df, embeddings

        for asin, group in df.groupby(asin_col):
            indices      = group.index.tolist()
            if len(indices) < 2:
                continue
            product_embs = embeddings[indices]
            sim_matrix   = product_embs @ product_embs.T
            np.fill_diagonal(sim_matrix, 0)
            n = len(indices)
            for i, idx in enumerate(indices):
                mean_sims[idx] = sim_matrix[i].sum() / max(n - 1, 1)
                max_sims[idx]  = sim_matrix[i].max()

        df["mean_cosine_similarity"] = mean_sims
        df["max_cosine_similarity"]  = max_sims
        logger.info("Embedding similarity features computed")
        return df, embeddings


# ─────────────────────────────────────────────────────────────
# Master Feature Builder
# ─────────────────────────────────────────────────────────────

class FakeReviewFeatureBuilder:

    def __init__(self):
        self.linguistic        = LinguisticFeatureExtractor()
        self.behavioural       = BehaviouralFeatureExtractor()
        self.embedding         = EmbeddingFeatureExtractor()
        self.linguistic_cols   = []
        self.behavioural_cols  = []
        self.embedding_cols    = ["mean_cosine_similarity", "max_cosine_similarity"]
        self.embeddings_matrix = None

    def build(self, df: pd.DataFrame, text_col: str = "clean_text") -> tuple:
       
        logger.info(f"Building features for {len(df)} reviews...")
        df = df.copy().reset_index(drop=True)

        ling_df = self.linguistic.extract_batch(df[text_col].tolist())
        self.linguistic_cols = ling_df.columns.tolist()
        
        duplicates = [c for c in ling_df.columns if c in df.columns]
        if duplicates:
            df = df.drop(columns=duplicates)
            
        df = pd.concat([df, ling_df], axis=1)

        df, self.behavioural_cols = self.behavioural.extract(df)

        embeddings             = self.embedding.encode(df[text_col].tolist())
        self.embeddings_matrix = embeddings
        df, _                  = self.embedding.compute_similarity_features(df, embeddings)

        all_cols       = self.linguistic_cols + self.behavioural_cols + self.embedding_cols
        feature_matrix = df[all_cols].fillna(0).values.astype(np.float32)

        logger.success(
            f"Feature matrix: {feature_matrix.shape} | "
            f"Linguistic: {len(self.linguistic_cols)} | "
            f"Behavioural: {len(self.behavioural_cols)} | "
            f"Embedding: {len(self.embedding_cols)}"
        )
        return feature_matrix, all_cols, df