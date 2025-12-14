import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    return cs, pl


@app.cell
def _(pl):
    tables = pl.read_csv("./bandi/*.csv")
    return (tables,)


@app.cell
def _(cs, pl, tables):
    (tables
        .with_columns(
            cs.string().str.to_lowercase()   
        )
        .filter(
            ~ (pl.col("value").eq("///") & pl.col("awarded_score").eq("esito sfavorevole")),
        )
        .filter(
            ~ (pl.col("value").eq("esito sfavorevole") & pl.col("awarded_score").is_null())
        )
        .filter(
            ~ pl.col("value").str.contains("^///?$")
        )
        .with_columns(
            must_pass = pl.when(
                pl.col("awarded_score").is_in(["esito sfavorevole", "non idoneo"])
            ).then(True)
        )
        .with_columns(
            pl.col("awarded_score").replace("esito sfavorevole", 0).replace("non idoneo", 0)
        )
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
