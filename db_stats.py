from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_parser.parser import get_dbstats_parser
from lib_report.jinja_environment import jinja_env, templates_dir

if TYPE_CHECKING:
    from collections.abc import Hashable


def main() -> int:
    """
    Generate report for given data analysis.

    Returns:
    --------
    int:
        Process exit status code:
        - 0: Success
        - 1: Error in file validation or loading
        - 2: Rendering or PDF generation error
    """
    # Get report parser
    parser = get_dbstats_parser()

    # Parse arguments
    args = parser.parse_args()

    # Parse arguments
    args = parser.parse_args()

    # Define file path
    filepath = Path("db/db.csv")

    # Read CSV file into DataFrame
    df = pd.read_csv(filepath)

    # Compute percentage of duplicates
    duplicated: float = round((df.duplicated().sum() / df.shape[0]) * (100), 2)

    # Define bin ages
    age_bins = [14, 29, 39, 49, 59, 69, 79]

    # Bin age
    df["age_binned"] = pd.cut(df["age"], bins=age_bins, right=True)

    # Compute summary table
    data: list[dict[Hashable, Any]] = (
        df.groupby(["test", "recruitment_year", "gender", "age_binned"], observed=True)
            .size()
            .reset_index(name="counts")
            .to_dict(orient="records")
    )

    # Load data
    try:
        # Get report template
        template = jinja_env.get_template("db_stats.html")

        # Build output paths
        base_path = Path("./data_out/db_stats")
        output_pdf = base_path.with_suffix(".pdf")
        output_html = base_path.with_suffix(".html")

        # Render HTML
        rendered_html: str =\
            template.render(data=data, header=args.header_letter, page=args.page_number)

        # Write HTML file
        with output_html.open("w") as fout:
            fout.write(rendered_html)

        # Write PDF file
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))
        print(f"Report generated: {output_pdf}")

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}")
        return 2

    print(f"Data loaded and validated successfully. Percentage of duplicates {duplicated}%")
    return 0

if __name__ == "__main__":
    sys.exit(main())

