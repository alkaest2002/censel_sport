from pathlib import Path

import orjson
import pandas as pd


def get_test_db() -> pd.DataFrame:
    """
    Generate report for given data analysis.
    """
    # Define file path
    db = pd.read_csv("db/db.csv")

    for file in Path("./data_out/").glob("**/*analysis.json"):
        data = orjson.loads(file.read_bytes())
        print(data, db.shape)
    return pd.DataFrame()


