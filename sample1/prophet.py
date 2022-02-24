from __future__ import annotations
from typing import Sequence, Tuple
from dataclasses import dataclass

from sklearn.metrics import mean_absolute_percentage_error
import prophet
import pandas as pd
from mdweek import Week

@dataclass
class TimeSeries:
    time: Sequence[Week]
    value: Sequence[float]

    def split(self, valid_start: Week) -> Tuple[TimeSeries, TimeSeries]:
        idx = 0
        while time[idx] < valid_start:
            idx += 1
        t1, t2 = self.time[:idx], self.time[idx:]
        v1, v2 = self.value[:idx], self.value[idx:]
        return TimeSeries(t1, v1), TimeSeries(t2, v2)


def test_prophet(data: TimeSeries, valid_start: Week, prophet_parameter: Dict[str, Any]) -> float:
    d1, d2 = data.split(valid_start)

    df1 = pd.DataFrame(dict(
        ds=[w.date(1) for w in d1.time],
        y=d1.value
    ))
    df2 = pd.DataFrame(dict(
        ds=[w.date(1) for w in d2.time],
        y=d2.value
    ))
    model = prophet.Prophet(**prophet_parameter)
    model.fit(df2)
    pred = model.predict(df2)

    return mean_absolute_percentage_error(df2.y, pred.yhat)

def test_arima(data: TimeSeries, valid_start: Week, arima_parameter: Dict[str, Any]) -> float:
    d1, d2 = data.split(valid_start)

    df1 = pd.DataFrame(dict(
        ds=[w.date(1) for w in d1.time],
        y=d1.value
    ))
    df2 = pd.DataFrame(dict(
        ds=[w.date(1) for w in d2.time],
        y=d2.value
    ))
    model = prophet.Prophet(**prophet_parameter)
    model.fit(df2)
    pred = model.predict(df2)

    return mean_absolute_percentage_error(df2.y, pred.yhat)
    
