"""news_fetcher.py

Fetches and converts Marketaux news data into numeric features usable by the ML pipeline.

Design goals:
- Never crash the main pipeline if news APIs fail.
- Return a fixed schema of numeric fields with safe defaults.
- Keep dependencies minimal: prefer `requests`.

Notes:
- Marketaux requires an API key.
- This module uses environment variable `MARKETAUX_API_KEY`.

"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, Optional

import requests


class NewsFetcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.marketaux.com/v1",
        timeout_s: int = 15,
        min_impact_abs: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("MARKETAUX_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.min_impact_abs = min_impact_abs

    def _default_news_features(self) -> Dict[str, float]:
        # Keep numeric-only schema for ML.
        return {
            "News_Sentiment_Avg": 0.0,  # [-1, 1] expected but we will clamp
            "News_Sentiment_Weighted": 0.0,
            "News_Count_24h": 0.0,
            "News_Impact_Sum": 0.0,
            "News_Recency_Weighted_Count": 0.0,
        }

    def fetch_news_features(self, ticker: str, company_name: Optional[str] = None) -> Dict[str, float]:
        """Fetch news for a ticker/company and return aggregated numeric features."""

        features = self._default_news_features()
        if not self.api_key:
            return features

        # Prefer ticker; fall back to company name query.
        query = ticker
        if not ticker and company_name:
            query = company_name
        if not query:
            return features

        # Marketaux: use /news?symbols=... (varies by plan). Some users use /news?symbols=ETH.
        # We try multiple endpoints defensively.
        endpoints = [
            (f"{self.base_url}/news", {"symbols": query}),
            (f"{self.base_url}/news", {"categories": query}),
            # If the plan supports it, some use /news?search=...
            (f"{self.base_url}/news", {"search": query}),
        ]

        headers = {}
        params_common = {"api_token": self.api_key, "limit": 50}

        all_news: list[Dict[str, Any]] = []
        for url, params in endpoints:
            params2 = dict(params_common)
            params2.update(params)
            try:
                resp = requests.get(url, params=params2, headers=headers, timeout=self.timeout_s)
                if resp.status_code != 200:
                    continue
                payload = resp.json() if resp.content else {}
                data = payload.get("data") or payload.get("news") or []
                if isinstance(data, list) and data:
                    all_news = data
                    break
            except Exception:
                continue

        if not all_news:
            return features

        # Aggregate. Marketaux response schema commonly has:
        # - sentiment: {polarity:..., sentiment:...} OR top-level 'sentiment'
        # - impact: numeric
        # - published_at: ISO timestamp
        # We'll handle multiple schema styles.
        now = time.time()

        sentiments = []
        weighted_sentiments = []
        impact_sum = 0.0
        impact_weighted_sent = 0.0

        count_24h = 0.0
        recency_weighted_count = 0.0

        for item in all_news:
            if not isinstance(item, dict):
                continue

            # Sentiment extraction
            sentiment_val = None
            if "sentiment" in item:
                sentiment_val = item.get("sentiment")
            elif isinstance(item.get("sentiment"), dict):
                sentiment_val = item.get("sentiment", {}).get("sentiment")
            elif isinstance(item.get("sentiment"), (int, float)):
                sentiment_val = item.get("sentiment")

            # Some schemas: 'sentiment' may be 0..100. We'll normalize if needed.
            if isinstance(sentiment_val, (int, float)):
                s = float(sentiment_val)
                if s > 1.5:  # likely 0..100 or 0..10
                    s = (s - 50.0) / 50.0
                sentiment_val = max(-1.0, min(1.0, s))

            # Impact extraction
            impact = 0.0
            raw_impact = item.get("impact")
            if isinstance(raw_impact, (int, float)):
                impact = float(raw_impact)

            published_at = item.get("published_at") or item.get("publishedAt") or item.get("date")
            age_hours = None
            if isinstance(published_at, str):
                try:
                    # Example: 2024-01-01T12:34:56Z
                    from datetime import datetime, timezone

                    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    age_hours = max(0.0, (now - dt.replace(tzinfo=timezone.utc).timestamp()) / 3600.0)
                except Exception:
                    age_hours = None

            if sentiment_val is None:
                sentiment_val = 0.0

            # Recency weights
            recency_w = 1.0
            if age_hours is not None:
                recency_w = 1.0 / (1.0 + age_hours / 12.0)  # decays over ~days

            if age_hours is not None and age_hours <= 24.0:
                count_24h += 1.0

            recency_weighted_count += recency_w

            sentiments.append(float(sentiment_val))
            weighted_sentiments.append(float(sentiment_val) * recency_w)

            if abs(impact) >= self.min_impact_abs:
                impact_sum += float(impact)
                impact_weighted_sent += float(sentiment_val) * recency_w * float(impact)

        if sentiments:
            features["News_Sentiment_Avg"] = float(sum(sentiments) / len(sentiments))

        if weighted_sentiments:
            denom = sum(1.0 for _ in weighted_sentiments) or 1.0
            features["News_Sentiment_Weighted"] = float(sum(weighted_sentiments) / denom)

        features["News_Count_24h"] = float(count_24h)
        features["News_Impact_Sum"] = float(impact_sum)
        features["News_Recency_Weighted_Count"] = float(recency_weighted_count)

        return features

