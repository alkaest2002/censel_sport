import sys
from typing import TYPE_CHECKING, Any

import pandas as pd

from lib_parser.parser import create_parser
from lib_parser.utils_parser import load_configuration_data, validate_file_path
from lib_report.utils_report import render_template

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

def main() -> int:
    """Generate report for given data analysis.

    Returns:
        int: Process exit status code:
            - 0: Success
            - 1: Error in file validation or loading
            - 2: Rendering or PDF generation error

    Raises:
        FileNotFoundError: If the input file or analysis script doesn't exist.
        ValueError: If the file path validation fails.
        Exception: For any other errors during report generation.
    """
    # Get report parser
    parser: argparse.ArgumentParser = create_parser(filepath=True)

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Validate the file path
    try:
        validated_path: Path = validate_file_path(args.filepath, "report")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    try:
        # Load and extratct data
        data: dict[str, Any] = load_configuration_data(validated_path)
        metric_id: str = data.get("metric_config", {}).get("metric_id", "unknown_metric")
        header_letter: str = data.get("metric_config", {}).get("report", {}).get("header_letter", "A")
        page_number: int = data.get("metric_config", {}).get("report", {}).get("initial_page", 1)

        # Render template with data
        output_paths: dict[str, Path] = render_template(
            jinja_template_name="report_annex.html",
            output_folder=validated_path.parent.parent / "_report",
            output_filename=f"{header_letter}_{metric_id}",
            output_formats=["pdf"],
            data=data,
            query_from_db=pd.DataFrame(data.get("query_from_db", {})),
            header=header_letter,
            page=page_number,
        )

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}", file=sys.stderr)
        return 2

    print(f"Report generated: {output_paths.get('pdf', 'No PDF generated')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
