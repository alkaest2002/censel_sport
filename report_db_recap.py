from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import orjson
import pandas as pd

from lib_parser.parser import create_parser
from lib_report.utils_report import render_template

if TYPE_CHECKING:
    import argparse

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

    # Globally find all analysis JSON files
    json_files = list(Path("./data_out").glob("**/*_analysis.json"))

    # Initialize new_norms dictionary
    new_norms: list[tuple[str, pd.DataFrame]] = []

    # Iterate over each JSON file
    for file in json_files:

        # Load JSON data
        with Path(file).open("r") as fin:
            data: dict[str, Any] = orjson.loads(fin.read())

        # Get metric config
        metric_config: dict[str, Any] = data["metric_config"]

        # Get test_id
        test_id: str = metric_config["test_id"]

        # Get metric_id
        metric_id: str = metric_config["metric_id"]

        # Get metric_type
        metric_type: str = metric_config["metric_type"]

        # Get metric_subtype
        metric_subtype: str = metric_config["metric_subtype"]

        # Get metric_precision
        metric_precision: int = metric_config["metric_precision"]

        # Get metric_precision_label
        metric_precision_label: str = metric_config["metric_precision_label"]

        # Get awarded scores
        awarded_scores: list[float] = metric_config["awarded_scores"]

        # Store bootstrap cutoffs
        cutoffs: list[list[float, float]] = data["bootstrap"]["cutoffs"]

        # Append norms
        new_norms.append(
            (
                metric_id,
                pd.concat(
                [
                    pd.DataFrame({
                        "test": [test_id] * len(cutoffs),
                        "metric_type": [metric_type] * len(cutoffs),
                        "metric_subtype": [metric_subtype] * len(cutoffs),
                        "metric_precision": [metric_precision] * len(cutoffs),
                        "metric_precision_label": [metric_precision_label]* len(cutoffs),
                        "gender": [metric_id.split("_")[-1][0].upper()] * len(cutoffs),
                    }),
                    pd.DataFrame(cutoffs, columns=["from", "to"]),
                    pd.Series(awarded_scores, name="awarded_score"),
                ], axis=1),
            ),
        )

        # sort norms
        sorted_norms: list[tuple[str, pd.DataFrame]] = sorted(new_norms, key=lambda x: x[0].split("_")[:-1])

    # Render template with data
    _: dict[str, Path] = render_template(
        jinja_template_name="report_db_recap.html",
        output_folder=Path("./data_out/_report"),
        output_filename=f"{args.header_letter}_db_recap",
        output_formats=["pdf"],
        data=sorted_norms,
        header=args.header_letter,
        page=args.page_number,
    )

    # Print success message
    print("Norms recap generated successfully.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
