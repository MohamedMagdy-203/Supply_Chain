from src.features import build_features, check_no_leakage


def test_build_features_has_no_leakage(weekly):
    assert check_no_leakage(build_features, weekly)


def test_leakage_test_rejects_leaky_feature(weekly):
    """Proof that the test can fail: rolling mean WITHOUT shift(1)."""
    def leaky(w):
        df = w.sort_values(["series_id", "week_idx"]).copy()
        df["origin_week"] = df["week_idx"] - 1
        df["roll_mean_4_leaky"] = df.groupby("series_id")["y"].transform(lambda s: s.rolling(4).mean())
        df["y_next"] = df["y"]
        return df[["series_id", "origin_week", "roll_mean_4_leaky", "y_next"]]

    assert check_no_leakage(leaky, weekly) is False


def test_origin_is_target_minus_one(weekly):
    f = build_features(weekly)
    assert (f["origin_week"] == f["week_idx"] - 1).all()


def test_lag1_is_previous_week_sales(weekly):
    f = build_features(weekly).set_index(["series_id", "week_idx"])
    w = weekly.set_index(["series_id", "week_idx"])["y"]
    sid, wk = f.index[100]
    assert f.loc[(sid, wk), "lag_1"] == w.loc[(sid, wk - 1)]


def test_no_duplicate_keys_and_sorted_splits(weekly):
    f = build_features(weekly)
    assert not f.duplicated(["series_id", "origin_week"]).any()
    assert f.loc[f.split == "train", "week_idx"].max() < f.loc[f.split == "val", "week_idx"].min()
    assert f.loc[f.split == "val", "week_idx"].max() < f.loc[f.split == "test", "week_idx"].min()