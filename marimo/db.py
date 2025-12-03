import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    db_original = pd.read_excel("./db/db.xlsx")
    return (db_original,)


@app.cell
def _(db_original):
    columns_mapper = { 
        "Anno_ril": "recruitment_year", 
        "Concorso": "recruitment_type", 
        "Prova": "test", 
        "Genere": "gender", 
        "Età": "age",
        "Performance": "value"
    }

    recrutiment_type_mapper = {
        "ACCADEMIA": "hd", 
        "Marescialli": "mlli"
    }

    test_mapper = {
        "100 m": "100mt_run",
        "1000 m": "1000mt_run",
        "Nuoto 25 m": "25mt_swim",
        "Pieg. addominali": "sit_ups",
        "Pieg. sulle braccia": "push_ups"
    }

    db_mapped = (
        db_original
            .rename(columns=columns_mapper)
            .loc[: , columns_mapper.values()]
            .replace({
                "recruitment_type": recrutiment_type_mapper,
                "test": test_mapper
            })
    )
    return db_mapped, test_mapper


@app.cell
def _(db_mapped):
    db_mapped.gender.eq("8")
    return


@app.cell
def _(db_mapped):
    db_cleaned = (
        db_mapped
            .loc[db_mapped.gender.ne("8"), :]
    )

    db_cleaned.head()
    return (db_cleaned,)


@app.cell
def _(db_cleaned):
    db_cleaned.recruitment_year.value_counts()
    return


@app.cell
def _(db_cleaned):
    db_cleaned.recruitment_type.value_counts()
    return


@app.cell
def _(db_cleaned, test_mapper):
    db_cleaned.test.value_counts().reindex(test_mapper.values())
    return


@app.cell
def _(db_cleaned):
    db_cleaned.gender.value_counts()
    return


@app.cell
def _(db_cleaned):
    db_cleaned.age.value_counts()
    return


@app.cell
def _(db_cleaned):
    db_cleaned.groupby(["test"])["value"].describe()
    return


@app.cell
def _(db_cleaned):
    db_cleaned.to_csv("./db/db.csv", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
