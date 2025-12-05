from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

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
    filepath: Path = Path("./db/new_norms_applied_to_db_results.csv")

    # Read db file into DataFrame
    data: pd.DataFrame = pd.read_csv(filepath)

    try:
        # Get db report template
        template: jinja2.Template = jinja_env.get_template("db_norms.html")

        # Build output path
        base_path: Path = Path(f"./data_out/_report/{args.header_letter}_db_norms")
        output_pdf: Path = base_path.with_suffix(".pdf")

        # Render template with data
        rendered_html: str =\
            template.render(
                tables_data=[
                    data.iloc[:39, :],
                    data.iloc[39:, :],
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
