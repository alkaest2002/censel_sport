from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import orjson
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
            - 1: Error in file validation or loading
            - 2: Rendering or PDF generation error
    """
    # Get report parser
    parser: argparse.ArgumentParser = get_db_report_parser()

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Get config files and parse them using orjson
    config_files: list[Path] = list(Path("./config").glob("*.json"))

    # Initialize TOC list
    toc: list[dict[str, Any]] = []

    # Iterate over config files and print their contents
    for config_file in config_files:
        config_data: dict[str, Any] = orjson.loads(config_file.read_text())
        toc.append({
            "title": config_data.get("title", "No Title"),
            "header_letter": config_data.get("report", {}).get("header_letter", "ZZZ"),
            "initial_page": config_data.get("report", {}).get("initial_page", 999),
        })

    # Reorder TOC by initial_page
    toc.sort(key=lambda x: x["initial_page"])

    # Load data
    try:
        # Get report template
        template: jinja2.Template = jinja_env.get_template("toc.html")

        # Build output paths
        base_path: Path = Path("./data_out/toc")
        output_pdf: Path = base_path.with_suffix(".pdf")
        output_html: Path = base_path.with_suffix(".html")

        rendered_html: str =\
            template.render(toc=toc, header=args.header_letter, page=args.page_number)

        # Write HTML file
        # Write HTML file
        with output_html.open("w") as fout:
            fout.write(rendered_html)

        # Write PDF file
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))
        print(f"TOC generated: {output_pdf}")

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}")
        return 2

    print("Toc was generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
