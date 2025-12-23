import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    return np, pd, pl


@app.cell
def _(pd):
    db = pd.read_csv("./db/db.csv")
    db.head()
    return (db,)


@app.cell
def _(db, np):
    pushups= db.query("""
    test.eq('push_ups')
    """)
    valid_mask = pushups.value.gt(0) & pushups.value.ne(np.isfinite)
    pushups[~valid_mask]
    return


@app.cell
def _(db, pl):
    dbl = pl.from_pandas(db)
    dbl.head()
    return (dbl,)


@app.cell
def _(dbl, pl):
    pushups_f_2022_pl = (
        dbl.filter(
            (pl.col("recruitment_year") == 2022) &
            (pl.col("recruitment_type") == "mlli") &
            (pl.col("test") == "push_ups") &
            (pl.col("gender") == "F")
        )
        .sort("value")
    )

    # Calculate the proportion of zeros
    pushups_f_2022_pl["value"].value_counts()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
