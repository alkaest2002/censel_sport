from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import orjson

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
    # Get report parser
    parser: argparse.ArgumentParser = create_parser(page_number=True)

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Get config files and parse them using orjson
    config_files: list[Path] = list(Path("./config").glob("*.json"))

    # Initialize TOC list
    toc: list[dict[str, Any]] = []

    # Iterate over config files
    for config_file in config_files:

        # Load and parse current config data
        config_data: dict[str, Any] = orjson.loads(config_file.read_text())

        # Append current config data to toc
        toc.append({
            "title": config_data.get("title", "No Title"),
            "header_letter": config_data.get("report", {}).get("header_letter", "ZZZ"),
            "initial_page": config_data.get("report", {}).get("initial_page", 999),
        })

    # Reorder TOC by initial_page
    toc.sort(key=lambda x: x["initial_page"])

    try:

        # Render template with data
        _: dict[str, Path] = render_template(
            jinja_template_name="report_toc.html",
            output_folder=Path("./data_out/_report"),
            output_filename="1_toc",
            output_formats=["pdf"],
            data={"toc": toc},
            page=args.page_number,
        )

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating TOC: {e}")
        return 1

    # Print success message
    print("TOC was generated successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
