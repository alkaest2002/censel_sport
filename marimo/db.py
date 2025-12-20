import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    return pd, pl


@app.cell
def _(pd):
    db = pd.read_csv("./db/db.csv")
    db.head()
    return (db,)


@app.cell
def _(db):
    pushups_f_2022 = db.query("""
        recruitment_year.eq(2022) & recruitment_type.eq('mlli') & test.eq('push_ups') & gender.eq('F')
    """).sort_values(by="value")
    pushups_f_2022.loc[:, ["value"]].eq(0).sum()/pushups_f_2022.shape[0]
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
    (pushups_f_2022_pl["value"] == 0).sum() / pushups_f_2022_pl.height
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
