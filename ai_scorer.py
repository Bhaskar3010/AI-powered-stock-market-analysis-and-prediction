"""Module 5 & 6: AI Scoring and Multi-Asset Ranking Pipeline Engine

Loads serialized artifacts and executes analytical processing arrays to score target instruments.

This implementation supports both:
- ML mode: uses `models/stock_model.pkl` via predict_proba
- Fallback mode: deterministic bounded scoring in [0, 100]

Feature schema (numeric):
- Technical/fundamental:
  RSI, MACD_Hist, BB_Position, SMA_Ratio, Volume_Trend,
  PE_Ratio, EPS_Growth, Market_Cap, Revenue_Growth, Debt_to_Equity
- News (Marketaux -> aggregated):
  News_Sentiment_Avg, News_Sentiment_Weighted, News_Count_24h,
  News_Impact_Sum, News_Recency_Weighted_Count
"""

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib


class AIScorer:
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), "models", "stock_model.pkl")
        else:
            self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """Loads the serialized model artifact or dynamic fallback parameters if needed."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None
        else:
            self.model = None

    def calculate_normalized_score(self, feature_dict: Dict[str, Any]) -> float:
        """Calculates a clean score bounded exactly within [0, 100]."""

        feature_order = [
            "RSI",
            "MACD_Hist",
            "BB_Position",
            "SMA_Ratio",
            "Volume_Trend",
            "PE_Ratio",
            "EPS_Growth",
            "Market_Cap",
            "Revenue_Growth",
            "Debt_to_Equity",
            # News features
            "News_Sentiment_Avg",
            "News_Sentiment_Weighted",
            "News_Count_24h",
            "News_Impact_Sum",
            "News_Recency_Weighted_Count",
        ]

        # ML mode
        if self.model is not None:
            try:
                safe_features = dict(feature_dict or {})
                for k in feature_order:
                    safe_features.setdefault(k, 0.0)

                input_df = pd.DataFrame([safe_features])[feature_order]
                probabilities = self.model.predict_proba(input_df)
                raw_score = float(probabilities[0][1]) * 100.0
                return float(np.clip(raw_score, 0.0, 100.0))
            except Exception:
                return 50.0

        # Fallback mode
        rsi = float(feature_dict.get("RSI", 50.0))
        rsi_factor = (100.0 - rsi) if rsi > 0 else 50.0

        pe = float(feature_dict.get("PE_Ratio", 25.0))
        pe_factor = 50.0 if pe <= 0 else max(10.0, 100.0 - pe)

        eps_growth = float(feature_dict.get("EPS_Growth", 0.05))

        news_sent = float(
            feature_dict.get(
                "News_Sentiment_Weighted",
                feature_dict.get("News_Sentiment_Avg", 0.0),
            )
        )
        news_sent_clamped = max(-1.0, min(1.0, news_sent))
        news_sent_score = (news_sent_clamped + 1.0) * 50.0  # [-1..1] -> [0..100]

        news_count = float(feature_dict.get("News_Count_24h", 0.0))
        news_impact = float(feature_dict.get("News_Impact_Sum", 0.0))

        # Keep news coverage from dominating.
        news_coverage_boost = min(20.0, news_count * 2.0)
        news_impact_boost = min(20.0, abs(news_impact) * 5.0)

        raw_composite = (
            (rsi_factor * 0.35)
            + (pe_factor * 0.25)
            + (eps_growth * 100.0 * 0.20)
            + (news_sent_score * 0.20)
            + news_coverage_boost * 0.05
            + news_impact_boost * 0.05
        )

        return float(np.clip(raw_composite, 0.0, 100.0))

