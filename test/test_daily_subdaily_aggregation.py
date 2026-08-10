import numpy as np
import pandas as pd

from code.daily_aggregation import (
    aggregate_eused_to_daily,
    aggregate_unix_series_to_daily,
    align_daily_series,
)


def test_eused_subdaily_recomputed_ssl_from_daily_means():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 06:00:00", "2020-01-02 00:00:00"]
            ),
            "Q": [1.0, 3.0, 5.0],
            "SSC": [10.0, 30.0, 50.0],
            "SSL": [100.0, 200.0, 300.0],
            "Q_derived": [False, False, False],
            "SSC_derived": [False, False, False],
            "SSL_derived": [False, False, False],
        }
    )

    out = aggregate_eused_to_daily(df)

    assert len(out) == 2
    assert out.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert np.isclose(out.loc[0, "Q"], 2.0)
    assert np.isclose(out.loc[0, "SSC"], 20.0)
    assert np.isclose(out.loc[0, "SSL"], 2.0 * 20.0 * 0.0864)
    assert bool(out.loc[0, "SSL_derived"])


def test_eused_duplicate_timestamp_and_ssl_only_day_are_preserved():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:00"]),
            "Q": [-9999.0, -9999.0],
            "SSC": [-9999.0, -9999.0],
            "SSL": [10.0, 20.0],
            "Q_derived": [False, False],
            "SSC_derived": [False, False],
            "SSL_derived": [False, False],
        }
    )

    out = aggregate_eused_to_daily(df)

    assert len(out) == 1
    assert np.isclose(out.loc[0, "SSL"], 15.0)
    assert out.loc[0, "Q"] == -9999.0
    assert out.loc[0, "SSC"] == -9999.0
    assert not bool(out.loc[0, "SSL_derived"])


def test_unix_subdaily_series_collapses_to_unique_utc_days():
    times = np.array([0.0, 3600.0, 7200.0, 86400.0])
    values = np.array([1.0, 3.0, 5.0, 10.0])

    days, daily_values = aggregate_unix_series_to_daily(times, values)

    assert np.array_equal(days, np.array([0.0, 86400.0]))
    assert np.allclose(daily_values, np.array([3.0, 10.0]))


def test_daily_alignment_uses_union_without_duplicate_days():
    union, aligned = align_daily_series(
        {
            "Q": (np.array([0.0, 86400.0]), np.array([1.0, 2.0])),
            "SSC": (np.array([86400.0, 172800.0]), np.array([10.0, 20.0])),
        }
    )

    assert np.array_equal(union, np.array([0.0, 86400.0, 172800.0]))
    assert np.isnan(aligned["Q"][2])
    assert np.isnan(aligned["SSC"][0])
    assert np.isclose(aligned["Q"][1], 2.0)
    assert np.isclose(aligned["SSC"][1], 10.0)
