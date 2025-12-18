import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_parser.parser import create_parser
from lib_parser.utils_parser import load_configuration_data, validate_file_path
from lib_report.jinja_environment import jinja_env, templates_dir

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    import jinja2

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
    # Render HTML
    render_html: bool = False

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

        # Get report template
        template: jinja2.Template = jinja_env.get_template("report_annex.html")

        # Render template
        rendered_html: str =\
            template.render(
                data=data,
                query_from_db=pd.DataFrame(data.get("query_from_db", {})),
                header=header_letter,
                page=page_number,
            )

        # Build output path
        output_pdf: Path =\
            (validated_path.parent.parent / "_report" / f"{header_letter}_{metric_id}").with_suffix(".pdf")

        # Write template to PDF file
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))

        if render_html:
            # Build HTML output path
            output_html: Path = validated_path.with_suffix(".html")

            # Write template to HTML file
            with output_html.open("w") as fout:
                fout.write(rendered_html)

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}", file=sys.stderr)
        return 2

    print(f"Report generated: {output_pdf}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
