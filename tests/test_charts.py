import pandas as pd

from datamoat.charts import feature_heatmap


def test_feature_heatmap_is_readable_and_explains_standardized_values():
    predictions = pd.DataFrame(
        {
            "name": ["Student A", "Student B"],
            "att_7": [0.60, 0.60],
            "score_overall": [0.52, 0.84],
        }
    )
    figure = feature_heatmap(
        predictions,
        ["att_7", "score_overall"],
        {
            "att_7": "Attendance in the last 7 days",
            "score_overall": "Overall test average",
        },
        limit=2,
    )

    assert figure is not None
    assert figure.layout.paper_bgcolor == "#FFFFFF"
    assert figure.layout.plot_bgcolor == "#FFFFFF"
    assert figure.layout.title.text == "Departure-risk signals vs institute average"
    assert list(figure.data[0].x) == ["Attend 7d", "Test average"]
    assert figure.data[0].customdata[0][0] == "Attendance in the last 7 days"
    assert "standard deviations from institute average" in figure.data[0].hovertemplate
    assert list(figure.data[0].z[:, 0]) == [0.0, 0.0]
