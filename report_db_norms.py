from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np
import orjson
import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_analysis.utils_stats import apply_standardization
from lib_parser.parser import create_parser
from lib_report.jinja_environment import jinja_env, templates_dir

if TYPE_CHECKING:
    import argparse

    import jinja2


def main() -> int:
    """Generate database statistics report.

    Returns:
        int: Process exit status code:
            - 0: Success
            - 1: Rendering or PDF generation error
    """
    # Get parser
    parser: argparse.ArgumentParser = create_parser(header_letter=True, page_number=True)

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Define db file path
    filepath: Path = Path("db/db.csv")

    # Read db file into DataFrame
    db: pd.DataFrame = pd.read_csv(filepath)

    # Aadd norms column
    db["norms"] = db["test"] + "_" + db["gender"].str.replace("M", "males").str.replace("F","females")

    # Initialize dictionary to hold norms data
    norms_dict: dict[str, any] = {}

    # Read analysis JSON files and populate norms_dict
    json_files = list(Path("./data_out").glob("**/*_analysis.json"))

    # Load norms data from JSON files
    for file in json_files:
        with Path(file).open("r") as fin:
            data = orjson.loads(fin.read())
            norms_dict[data["metric_config"]["id"]] = data["bootstrap"]["cutoffs"]

    # Initialize results dictionary
    results = {}

    # Process each group in the database
    for group_label, group_data in db.groupby(["test", "recruitment_type", "recruitment_year", "gender"]):
        # Create a unique key for the group
        key = "_".join(map(str, group_label))
        # Extract values to standardize
        values = group_data["value"].to_numpy()
        # Get cutoffs for the group's norms
        cutoffs = norms_dict.get(group_data.iloc[0].loc["norms"])
        # Apply standardization and compute step distribution
        results[key] = (
            apply_standardization(data_to_standardize=values, cutoffs=cutoffs)["standardized_step"]
                .value_counts(normalize=True, sort=False)
                .mul(100)
                .round(1)
                .sort_index()
                .add_prefix("step_")
                .to_dict()
        )

    # Convert results to DataFrame
    results_df: pd.DataFrame = pd.DataFrame(results).fillna(0).T

    # Prepare report data
    report_data: pd.DataFrame = (
        pd.DataFrame(
            np.hstack(
                [
                    pd.Series(results_df.index).str.rsplit("_", n=3, expand=True).to_numpy(),
                    results_df.to_numpy(),
                ],
            ),
            columns=["test","recruitment_type","recruitment_year","gender","step1","step2","step3","step4","step5","step6"],
        )
        .replace({ "recruitment_type": {"hd": "Accademia", "mlli": "Marescialli"} })
    )

    try:
        # Get db report template
        template: jinja2.Template = jinja_env.get_template("report_db_norms.html")

        # Build output path
        base_path: Path = Path(f"./data_out/_report/{args.header_letter}_db_norms")
        output_pdf: Path = base_path.with_suffix(".pdf")

        # Render template with data
        rendered_html: str =\
            template.render(
                tables_data=[
                    report_data.iloc[:39, :],
                    report_data.iloc[39:, :],
                ],
                header=args.header_letter,
                page=args.page_number,
            )

        # Write PDF file
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))
        print(f"Report generated: {output_pdf}")

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
