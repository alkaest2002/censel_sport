import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import locale
    from itertools import product
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    locale.setlocale(locale.LC_ALL, "it_IT.UTF-8")
    return pd, product


@app.cell
def _(pd):
    sheet_name = "100m_M"
    sheet_id = "1YM331ADxGwKQbC_d7FM35FPhoiImVV7x"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    tables = pd.read_csv(url, usecols=range(8))
    tables.head()
    return (tables,)


@app.cell
def _(pd, product):
    def reshape_table(tables, is_time=False):
        df = (
            tables
                .drop(["Nuova tabella FASCIA", "PUNTEGGIO.1"], axis=1)
                .rename(columns={
                    "Vecchia tabella FASCIA": "fascia",
                    "LIMITE INFERIORE": "inf",
                    "LIMITE INFERIORE.1": "inf",
                    "LIMITE SUPERIORE": "sup", 
                    "LIMITE SUPERIORE.1": "sup",
                    "PUNTEGGIO": "awarded"
                })
                .assign(awarded=lambda df: df.awarded.str.replace(",", ".").astype(float))
                .set_index("fascia")
        )
        c = pd.concat([df.iloc[:, [0,1,2]], df.iloc[:, [0,3,4]]])
    
        def parse_time_to_seconds(time_str):
            """Convert MM:SS.T directly to total seconds"""
            if pd.isna(time_str):
                return None
            if not isinstance(time_str, str):
                return float(time_str)
        
            if ':' in time_str and '.' in time_str:
                minutes, seconds_tenths = time_str.split(':')
                seconds, tenths = seconds_tenths.split('.')
                return int(minutes) * 60 + int(seconds) + int(tenths) / 10
            return float(time_str)
    
        if is_time:
            # Convert directly to seconds (no timedelta)
            c.inf = c.inf.apply(parse_time_to_seconds)
            c.sup = c.sup.apply(parse_time_to_seconds)
   
    
        c.index = pd.MultiIndex.from_tuples(
            product(("old", "new"), range(1,7)), 
            names=["type_of_table", "band"]
        )
        return c.reset_index(drop=False)

    return (reshape_table,)


@app.cell
def _(reshape_table, tables):
    t = reshape_table(tables, True)
    t.pivot_table(index=["band","awarded"], columns=["type_of_table"]).swaplevel(axis=1).sort_index(axis=1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
