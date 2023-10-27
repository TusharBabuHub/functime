from typing import Optional

import polars as pl

from functime.preprocessing import lag


def _join_X_y(y: pl.LazyFrame, X: pl.LazyFrame) -> pl.LazyFrame:
    on = set(y.columns[:2]) & set(X.columns[:2])
    if len(on) < 1:
        raise ValueError(
            "`X` must have at least one leading column identical to `y`'s"
            f" leading columns ({y.columns[0]}, {y.columns[1]})."
        )
    return y.join(X, on=on, how="inner")


def make_reduction(
    lags: int, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None
) -> pl.DataFrame:
    idx_cols = y.columns[:2]
    # Defensive lazy
    y = y.lazy()
    X = X.lazy() if X is not None else X
    # Get lags
    y_lag = y.pipe(lag(lags=list(range(1, lags + 1))))
    X_y = y_lag.join(y, on=idx_cols, how="inner").select(
        [*y.columns, *y_lag.columns[2:]]
    )
    # Exogenous features
    if X is not None:
        X_y = _join_X_y(X_y, X)
    return X_y.collect()


def make_direct_reduction(
    lags: int, max_horizons: int, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None
) -> pl.DataFrame:
    idx_cols = y.columns[:2]
    # Defensive lazy
    y = y.lazy()
    X = X.lazy() if X is not None else X
    # Get lags
    y_lag = y.pipe(lag(lags=list(range(1, lags + max_horizons + 1))))
    X_y = y_lag.join(y, on=idx_cols, how="inner").select(
        [*y.columns, *y_lag.columns[2:]]
    )
    # Drop nulls in lagged columns
    if X is not None:
        X_y = _join_X_y(X_y, X)
    return X_y.collect()


def make_y_lag(X_y: pl.DataFrame, target_col: str, lags: int):
    # NOTE: We should probably do a defensive sort on time before concat_list
    entity_col, time_col = X_y.columns[:2]
    return (
        X_y.lazy()
        .select([entity_col, time_col, pl.col(rf"^{target_col}__lag_(\d+)$")])
        .group_by(entity_col)
        .agg(pl.all().tail(lags))
        .collect(streaming=True)
        .lazy()
    )
