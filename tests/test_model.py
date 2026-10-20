from ai_scorer import AIScorer


def test_scorer_boundary_constraints():
    scorer = AIScorer(model_path="non_existent_path.pkl")
    scorer.load_model()  # Trigger model failure gracefully

    mock_features = {
        "RSI": 110.0,  # anomaly value boundary checks
        "PE_Ratio": -5.0,
        "EPS_Growth": 0.25,
        "Market_Cap": 1e12,
        # news fields present so fallback doesn't crash
        "News_Sentiment_Avg": -0.9,
        "News_Sentiment_Weighted": -0.9,
        "News_Count_24h": 0.0,
        "News_Impact_Sum": -2.0,
        "News_Recency_Weighted_Count": 0.0,
    }

    score = scorer.calculate_normalized_score(mock_features)
    assert 0.0 <= score <= 100.0

