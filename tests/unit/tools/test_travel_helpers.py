from mini_devin.tools.browser.travel_helpers import format_hotel_summary, summarize_hotel_options


def test_summarize_hotel_options_picks_cheapest_best_rated_and_best_value():
    summary = summarize_hotel_options(
        [
            {
                "name": "Budget Stay",
                "price_text": "$120",
                "rating": "4.1",
                "review_count": "120",
                "text": "Budget Stay $120 4.1/5 120 reviews",
            },
            {
                "name": "Luxury View",
                "price_text": "$260",
                "rating": "4.9",
                "review_count": "980",
                "free_cancellation": True,
                "text": "Luxury View $260 4.9/5 980 reviews free cancellation",
            },
            {
                "name": "Value Suites",
                "price_text": "$170",
                "rating": "4.7",
                "review_count": "640",
                "free_cancellation": True,
                "breakfast_included": True,
                "text": "Value Suites $170 4.7/5 640 reviews free cancellation breakfast included",
            },
        ]
    )

    assert summary is not None
    assert summary["cheapest"]["name"] == "Budget Stay"
    assert summary["best_rated"]["name"] == "Luxury View"
    assert summary["best_value"]["name"] == "Value Suites"
    assert summary["count"] == 3


def test_format_hotel_summary_mentions_detected_options():
    summary = summarize_hotel_options(
        [
            {
                "name": "City Hotel",
                "price_text": "$140",
                "rating": "4.4",
                "review_count": "210",
                "text": "City Hotel $140 4.4/5 210 reviews",
            },
            {
                "name": "Bay Resort",
                "price_text": "$190",
                "rating": "4.8",
                "review_count": "410",
                "free_cancellation": True,
                "text": "Bay Resort $190 4.8/5 410 reviews free cancellation",
            },
        ]
    )

    assert summary is not None
    text = format_hotel_summary(summary)
    assert "Detected 2 hotel-like options." in text
    assert "Cheapest:" in text
    assert "Best rated:" in text
