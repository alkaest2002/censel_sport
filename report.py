"""CLI utility to generate a statistical analysis report from a data file.

This module validates an input file path supplied via the command line,
loads the JSON data, renders a Jinja2 HTML template, and converts it to
a PDF using WeasyPrint. It is intended to be used as an executable script.

Example:
    $ python report.py path/to/data.json
"""

from pathlib import Path
import sys
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_analysis.utils_generic import format_seconds
from lib_parser.parser import get_base_parser
from lib_parser.utils_parser import load_configuration_data, validate_file_path


def main() -> int:
    """
    Generate report for given data analysis.

    int
        Process exit status code:
        - 0: Success
        - 1: Error in file validation or loading
        - 2: Rendering or PDF generation error
    """
    # Parse command line arguments
    parser = get_base_parser()

    # Add parser argument numbering for report generation
    parser.add_argument(
        "--header-numbering", "-x",
        required=True,
        type=str,
        help="Numbering style for report header section (e.g., 'A1', 'A2')",
    )

    parser.add_argument(
        "--page-numbering-start", "-p",
        required=False,
        type=int,
        help="Starting page number for report pages (e.g., 1)",
    )

    args = parser.parse_args()

    # Validate the file path
    try:
        validated_path = validate_file_path(args.filepath, "report")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    # Init jinja environment
    templates_dir = Path("./lib_report").resolve()
    jinja_env: Environment = Environment(
        # Use FileSystemLoader to load templates from the 'report' directory
        loader=FileSystemLoader(str(templates_dir)),
        # Enable strict undefined handling to catch missing variables
        undefined=StrictUndefined,
        # Auto-escape HTML for security
        autoescape=select_autoescape(["html", "xml"]),
    )

    jinja_env.filters["format_seconds"] = format_seconds

    # Load data and render template, then generate PDF
    try:
        data: dict[str, Any] = load_configuration_data(validated_path)
        template = jinja_env.get_template("report.html")
        output_pdf = validated_path.with_suffix(".pdf")
        output_html = validated_path.with_suffix(".html")
        header_numbering = list(map(str.strip, args.header_numbering.split(","))) if args.header_numbering else []
        page_numbering_start = args.page_numbering_start if args.page_numbering_start else 1
        rendered_html: str =\
            template.render(data=data, header_numbering=header_numbering, page_numbering_start=page_numbering_start)

        # Write html
        with output_html.open("w") as fout:
            fout.write(rendered_html)

        # Write PDF
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))
        print(f"Report generated: {output_pdf}")

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
