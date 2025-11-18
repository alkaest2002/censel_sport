from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_parser.parser import get_db_report_parser
from lib_report.jinja_environment import jinja_env, templates_dir

if TYPE_CHECKING:
    import argparse

    import jinja2


def _stringify_value_counts(x: pd.Series) -> str:
    """Convert a dictionary to a string representation.

    Args:
        x: Series to convert.

    Returns:
        String representation of the dictionary.
    """
    # Collect value counts as dictionary
    value_counts_dict: dict[Any, int] = x.value_counts().to_dict()
    # Omit keys with values equal to zero
    value_counts_dict = {k: v for k, v in value_counts_dict.items() if v != 0}
    # Convert to string with custom formatting
    return (
        str(value_counts_dict)[1:-1]
        .replace("'", "")
        .replace(",", " | ")
        .replace(":", " &rarr; ")
    )

def main() -> int:
    """Generate database statistics report.

    This function reads a CSV database file, processes the data to compute
    statistics including duplicates percentage and age-binned summaries,
    then generates both HTML and PDF reports using Jinja templates.

    Returns:
        int: Process exit status code:
            - 0: Success
            - 1: Rendering or PDF generation error
    """
    # Get report parser
    parser: argparse.ArgumentParser = get_db_report_parser()

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Define file path
    filepath: Path = Path("db/db.csv")

    # Read CSV file into DataFrame
    df: pd.DataFrame = pd.read_csv(filepath)

    # Compute percentage of duplicates
    duplicated: float = round((df.duplicated().sum() / df.shape[0]) * (100), 2)

    # Define bin ages
    age_bins: list[int] = [13, 29, 39, 49, 59, 69, 79]

    # Bin age
    df["age_binned"] = pd.cut(
        df["age"], bins=age_bins,
        labels=["17-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
        right=True,
    )

    # Group by test and recruitment year
    grouped = df.groupby(["test", "recruitment_year"])

    # Create data for report
    data = (pd.concat([
        grouped["gender"].apply(_stringify_value_counts),
        grouped["age_binned"].apply(_stringify_value_counts),
        grouped.size().rename("counts"),
        ], axis=1).reset_index(names=["test", "recruitment_year"]))

    # Load data
    try:
        # Get report template
        template: jinja2.Template = jinja_env.get_template("db_stats.html")

        # Build output paths
        base_path: Path = Path("./data_out/_report/A_db_stats")
        output_pdf: Path = base_path.with_suffix(".pdf")

        rendered_html: str =\
            template.render(data=data, header=args.header_letter, page=args.page_number)

        # Write PDF file
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))
        print(f"Report generated: {output_pdf}")

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}", file=sys.stderr)
        return 1

    print(f"Data loaded and validated successfully. Percentage of duplicates {duplicated}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
