from __future__ import annotations
from typing import Any, Dict, Sequence, Tuple
from dataclasses import dataclass

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import prophet
import pandas as pd
from mdweek import Week
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt


@dataclass
class TimeSeries:
    time: Sequence[Week]
    value: Sequence[float]

    def split(self, valid_start: Week) -> Tuple[TimeSeries, TimeSeries]:
        idx = 0
        while self.time[idx] < valid_start:
            idx += 1
        t1, t2 = self.time[:idx], self.time[idx:]
        v1, v2 = self.value[:idx], self.value[idx:]
        return TimeSeries(t1, v1), TimeSeries(t2, v2)


@dataclass
class Result:
    df: pd.DataFrame
    mape: float
    mse: float


def test_prophet(data: TimeSeries, valid_start: Week, prophet_parameter: Dict[str, Any]) -> Result:
    d1, d2 = data.split(valid_start)

    df1 = pd.DataFrame(dict(
        ds=[w.date(1) for w in d1.time],
        y=d1.value
    ))
    model = prophet.Prophet(**prophet_parameter)
    model.fit(df1)
    ret1 = pd.DataFrame(dict(
        week=d1.time,
        y=d1.value,
        yhat=model.predict()
    ))
    ret2 = pd.DataFrame(dict(
        week=d2.time,
        y=d2.value,
        yhat=model.predict(df2)
    ))

    return Result(
        df=pd.concat([ret1, ret2]),
        mape=mean_absolute_percentage_error(ret2.y, ret2.yhat),
        mse=mean_squared_error(ret2.y, ret2.yhat)
    )


def test_arima(data: TimeSeries, valid_start: Week, arima_parameter: Dict[str, Any]) -> Result:
    d1, d2 = data.split(valid_start)

    model = ARIMA(d1.value, **arima_parameter)
    param = model.fit()

    n1 = len(d1.value)
    n2 = len(d2.value)
    yhat1 = model.predict(param, start=1, end=n1)
    yhat2 = model.predict(param, start=n1+1, end=n1+n2)

    ret1 = pd.DataFrame(dict(
        week=d1.time,
        y=d1.value,
        yhat=yhat1
    ))
    ret2 = pd.DataFrame(dict(
        week=d2.time,
        y=d2.value,
        yhat=yhat2
    ))

    return Result(
        df=pd.concat([ret1, ret2]),
        mape=mean_absolute_percentage_error(ret2.y, ret2.yhat),
        mse=mean_squared_error(ret2.y, ret2.yhat)
    )
