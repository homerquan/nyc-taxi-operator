import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mobility.evaluation import metric_summary


def test_streamed_metrics_match_tie_aware_reference(tmp_path):
    frame = pd.DataFrame({"actual_trips":[1.,0.,2.,0.], "predicted_trips":[1.,.2,1.5,.1],
                          "link_probability":[.7,.7,.8,.2], "actual_duration":[10.,0.,20.,0.],
                          "predicted_duration":[11.,0.,18.,0.], "actual_fare":[20.,0.,30.,0.],
                          "predicted_fare":[19.,0.,33.,0.], "departure_tvd":[.2,0.,.4,0.]})
    path = tmp_path / "evaluation.parquet"
    frame.to_parquet(path)
    metrics = metric_summary(path, [1.,2.])
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(frame.actual_trips > 0, frame.link_probability))
    assert metrics["average_precision"] == pytest.approx(average_precision_score(frame.actual_trips > 0, frame.link_probability))
    assert metrics["duration_mae"] == pytest.approx(1.5)
    assert metrics["mrr"] == pytest.approx(.75)
