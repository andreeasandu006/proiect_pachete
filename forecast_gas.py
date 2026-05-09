import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor
import holidays


def load_data_from_csv():
    path = "date"
    df_hist    = pd.read_csv(os.path.join(path, "gas_history.csv"))
    df_temp_f  = pd.read_csv(os.path.join(path, "gas_temp_forecast.csv"))
    df_cloud_f = pd.read_csv(os.path.join(path, "gas_cloud_cover_forecast.csv"))

    # Istoric: format zz-ll-aa
    df_hist["date"] = pd.to_datetime(df_hist["record_date"], format="%d-%m-%y")
    df_hist = df_hist.rename(columns={"temp_daily": "temperature", "cloud_cover_daily": "cloud"})
    df_hist = df_hist.drop(columns=["record_date"])

    # Forecast: format ISO standard
    df_temp_f  = df_temp_f.rename(columns={"target_date": "date", "temp_daily_forecast": "temperature"})
    df_cloud_f = df_cloud_f.rename(columns={"target_date": "date", "cloud_cover_daily_forecast": "cloud"})
    df_temp_f["date"]  = pd.to_datetime(df_temp_f["date"])
    df_cloud_f["date"] = pd.to_datetime(df_cloud_f["date"])

    for df in [df_hist, df_temp_f, df_cloud_f]:
        df.set_index("date", inplace=True)
        df.index = df.index.normalize()

    df_hist    = df_hist[~df_hist.index.duplicated(keep="last")]
    df_temp_f  = df_temp_f[~df_temp_f.index.duplicated(keep="last")]
    df_cloud_f = df_cloud_f[~df_cloud_f.index.duplicated(keep="last")]

    for col in ["temperature", "cloud", "gas_consumption"]:
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].fillna(df_hist[col].median())

    return df_hist, df_temp_f, df_cloud_f


def add_holiday_features(index):
    ro_holidays = holidays.Romania()
    return pd.DataFrame({
        "is_holiday":        [1 if d in ro_holidays else 0 for d in index],
        "is_before_holiday": [1 if (d + pd.Timedelta(days=1)) in ro_holidays else 0 for d in index],
        "is_after_holiday":  [1 if (d - pd.Timedelta(days=1)) in ro_holidays else 0 for d in index],
    }, index=index)


def train_and_forecast():
    df_hist, df_temp_f, df_cloud_f = load_data_from_csv()
    df_hist = df_hist.sort_index()

    t_mean = df_hist["temperature"].mean()
    t_std  = df_hist["temperature"].std()
    t_norm = (df_hist["temperature"] - t_mean) / t_std
    h_val  = np.maximum(18 - df_hist["temperature"], 0)

    x_train = pd.DataFrame({
        "temp_norm":  t_norm,
        "temp_lag1":  t_norm.shift(1),
        "temp_lag7":  t_norm.shift(7),
        "hdd":        h_val,
        "hdd_lag1":   h_val.shift(1),
        "hdd_lag7":   h_val.shift(7),
        "cloud":      df_hist["cloud"],
        "cloud_lag1": df_hist["cloud"].shift(1),
        "gas_lag1":   df_hist["gas_consumption"].shift(1),
        "gas_lag7":   df_hist["gas_consumption"].shift(7),
        "gas_roll7":  df_hist["gas_consumption"].shift(1).rolling(7).mean(),
        "dow":        df_hist.index.dayofweek.astype(float),
        "month":      df_hist.index.month.astype(float),
    })

    hols_hist = add_holiday_features(df_hist.index)
    x_train   = pd.concat([x_train, hols_hist], axis=1)

    valid_idx = x_train.dropna().index
    x_train   = x_train.loc[valid_idx]
    y_train   = df_hist["gas_consumption"].loc[valid_idx]

    mask    = y_train.notna()
    x_train = x_train.loc[mask]
    y_train = y_train.loc[mask]

    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.03, max_depth=3, random_state=42
    )
    model.fit(x_train, y_train)

    future_dates = df_temp_f.index.intersection(df_cloud_f.index).sort_values()

    temp_all      = pd.concat([df_hist["temperature"], df_temp_f["temperature"]])
    cloud_all     = pd.concat([df_hist["cloud"],       df_cloud_f["cloud"]])
    hdd_all       = np.maximum(18 - temp_all, 0)
    temp_norm_all = (temp_all - t_mean) / t_std
    hols_future   = add_holiday_features(future_dates)

    y_series      = df_hist["gas_consumption"].copy()
    preds         = []
    features_list = x_train.columns.tolist()

    t_norm_s1 = temp_norm_all.shift(1)
    t_norm_s7 = temp_norm_all.shift(7)
    hdd_s1    = hdd_all.shift(1)
    hdd_s7    = hdd_all.shift(7)
    cloud_s1  = cloud_all.shift(1)

    def safe_get(series, key, fallback):
        try:
            val = series.loc[key]
            return float(val) if pd.notna(val) else fallback
        except KeyError:
            return fallback

    for dt in future_dates:
        row = {
            "temp_norm":         safe_get(temp_norm_all, dt, 0.0),
            "temp_lag1":         safe_get(t_norm_s1,     dt, 0.0),
            "temp_lag7":         safe_get(t_norm_s7,     dt, 0.0),
            "hdd":               safe_get(hdd_all,       dt, 0.0),
            "hdd_lag1":          safe_get(hdd_s1,        dt, 0.0),
            "hdd_lag7":          safe_get(hdd_s7,        dt, 0.0),
            "cloud":             safe_get(cloud_all,     dt, float(cloud_all.median())),
            "cloud_lag1":        safe_get(cloud_s1,      dt, float(cloud_all.median())),
            "gas_lag1":          float(y_series.iloc[-1]),
            "gas_lag7":          float(y_series.iloc[-7]) if len(y_series) >= 7 else float(y_series.mean()),
            "gas_roll7":         float(y_series.iloc[-7:].mean()),
            "dow":               float(dt.dayofweek),
            "month":             float(dt.month),
            "is_holiday":        float(hols_future.loc[dt, "is_holiday"]),
            "is_before_holiday": float(hols_future.loc[dt, "is_before_holiday"]),
            "is_after_holiday":  float(hols_future.loc[dt, "is_after_holiday"]),
        }
        x_input = pd.DataFrame([row])[features_list]
        y_hat   = model.predict(x_input)[0]
        preds.append(y_hat)
        y_series.loc[dt] = y_hat

    df_prog = pd.DataFrame({"forecast": preds}, index=future_dates)
    return df_prog, model, x_train, y_train, features_list


if __name__ == "__main__":
    res, _, _, _, _ = train_and_forecast()
    print(res)
    res.to_csv("date/gas_predictions_output.csv")