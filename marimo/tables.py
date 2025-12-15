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
    clean1 = (tables
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
            pl.col("awarded_score")
                .replace(["esito sfavorevole", "non idoneo"], 0)
                .str.replace(r"\,", ".")
                .cast(pl.Float64)
        )
    )
    clean1
    return (clean1,)


@app.cell
def _(clean1):
    df = clean1.to_pandas()
    df["value_from"] = None
    df["value_to"] = None
    return (df,)


@app.cell
def _(df):
    df.loc[df.awarded_score.eq(0), "value_from"] = df.value

    p1 = r"^da (.+) a "
    c1 = df.value.str.contains(p1)
    e1 = df.value.str.extract(p1, expand=False)
    df.loc[c1, "value_from"] = e1

    p2 = r"^(.+) –"
    c2 = df.value.str.contains(p2)
    e2 = df.value.str.extract(p2, expand=False)
    df.loc[c2, "value_from"] = e2

    # Review everything left that is not 1 or 2
    df.loc[df.value_from.isna() & ~df.awarded_score.isin([1,2]), :]
    return


@app.cell
def _(df):
    p3 = r"^oltre (.+)"
    c3 = df.value.str.contains(p3)
    e3 = df.value.str.extract(p3, expand=False)
    df.loc[c3, "value_from"] = e3

    p4 = r"^>(.+)"
    c4 = df.value.str.contains(p4)
    e4 = df.value.str.extract(p4, expand=False)
    df.loc[c4, "value_from"] = e4

    p5 = r"^inferiore a (.+)"
    c5 = df.value.str.contains(p5)
    e5 = df.value.str.extract(p5, expand=False)
    df.loc[c5, "value_to"] = e5

    p6 = r"a (.+)$"
    c6 = df.value.str.contains(p6)
    e6 = df.value.str.extract(p6, expand=False)
    df.loc[c6, "value_to"] = e6

    p7 = r"– (.+)$"
    c7 = df.value.str.contains(p7)
    e7 = df.value.str.extract(p7, expand=False)
    df.loc[c7, "value_to"] = e7


    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
