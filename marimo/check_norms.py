import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import polars as pl
    import orjson
    return Path, np, orjson, pd


@app.cell
def _(Any, NDArray, np, pd):
    def apply_standardization(
            data_to_standardize: NDArray[np.number[Any]],
            cutoffs: list[tuple[float, float]],
        ) -> pd.DataFrame:
        """Apply standardization to data using percentile cutoffs.

        Args:
            data_to_standardize: Numerical data array to be standardized.
            cutoffs: List of tuples containing percentile cutoff ranges.

        Returns:
            pd.DataFrame: DataFrame with standadization results.
        """
        # Convert data to pandas Series for easier manipulation
        data: pd.Series = pd.Series(data_to_standardize)

        # Add inclusive bounds to cutoffs
        # First cutoff is inclusive on both sides, others only on the right
        cutoffs_with_inclusive: list[tuple[tuple[float, float], str]] = \
            list(zip(cutoffs, ["both", *["right"] * (len(cutoffs) - 1)], strict=True))

        # Compute standardized steps
        standardized_steps: pd.Series = data.case_when(
            [
                (lambda x, cutoffs=cutoffs, inclusive=inclusive:
                    x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), i)
                for i, (cutoffs, inclusive) in enumerate(cutoffs_with_inclusive, start=1)
            ],
        ).astype(int)

        return pd.concat(
            [
                data,
                standardized_steps,
            ],
            keys=[
                "original_value",
                "standardized_step",
                ],
            axis=1,
        )
    return (apply_standardization,)


@app.cell
def _(pd):
    db = pd.read_csv("./db/db.csv")
    db["gender"] = db["gender"].str.replace("M", "males").str.replace("F","females")
    db["norms"] = db["test"] + "_" + db["gender"]
    db.head()
    return (db,)


@app.cell
def _(Path, orjson):
    norms_dict = {}
    json_files = list(Path("./data_out").glob("**/*_analysis.json"))

    for file in json_files:
        with Path(file).open("r") as fin:
            data = orjson.loads(fin.read())
            norms_dict[data["metric_config"]["id"]] = data["bootstrap"]["cutoffs"]
    return (norms_dict,)


@app.cell
def _(apply_standardization, db, norms_dict, pd):
    results = {}

    for group_label, group_data in db.groupby(["test", "recruitment_type", "recruitment_year", "gender"]):
        key = "_".join(map(str,group_label))
        values = group_data["value"].to_numpy()
        cutoffs = norms_dict.get(group_data.iloc[0].loc["norms"])
        results[key] = (
            apply_standardization(data_to_standardize=values, cutoffs=cutoffs)["standardized_step"]
                .value_counts(normalize=True, sort=False)
                .mul(100)
                .round(1)
                .sort_index()
                .add_prefix("step_")
                .to_dict()
        )

    results_df = pd.DataFrame(results).fillna(0).T
    results_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
