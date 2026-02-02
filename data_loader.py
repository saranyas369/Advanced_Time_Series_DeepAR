import pandas as pd
import numpy as np
import holidays

def load_multivariate_data(n=500):
    rng = np.random.default_rng(42)
    time = pd.date_range("2022-01-01", periods=n, freq="D")

    data = pd.DataFrame({
        "date": time,
        "target": np.sin(np.arange(n)/20) + rng.normal(0,0.3,n),
        "temp": rng.normal(25,5,n),
        "promotion": rng.integers(0,2,n)
    })

    data["lag_1"] = data["target"].shift(1)
    data["lag_7"] = data["target"].shift(7)

    ind_holidays = holidays.India()
    data["holiday"] = data["date"].isin(ind_holidays).astype(int)

    return data.dropna()