"""CLI utility to generate a statistical analysis report from a data file.

This module validates an input file path supplied via the command line,
loads the JSON data, renders a Jinja2 HTML template, and converts it to
a PDF using WeasyPrint. It is intended to be used as an executable script.

Example:
    $ python report.py path/to/data.json
"""
from pathlib import Path
import subprocess
import sys
from typing import Any

from weasyprint import HTML  # type: ignore[import-untyped]

from lib_parser.parser import get_report_parser
from lib_parser.utils_parser import load_configuration_data, validate_file_path
from lib_report.jinja_environment import jinja_env, templates_dir


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
    parser = get_report_parser()

    # Parse arguments
    args = parser.parse_args()

    # Validate the file path
    try:
        validated_path = validate_file_path(args.filepath, "report")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    # Re-run analysis, if requested
    if args.recompute:

        # Resolve path to analysis script
        analysis_script = Path("inner_loop.py").resolve()

        # Validate that the analysis script exists and is safe to execute
        if not analysis_script.exists():
            print(f"Error: Analysis script not found: {analysis_script}")
            return 1

        # Execute the analysis script
        subprocess.run([sys.executable, str(analysis_script), "-f", args.filepath], check=True)

    try:
        # Load data
        data: dict[str, Any] = load_configuration_data(validated_path)

        # Get report template
        template = jinja_env.get_template("report.html")

        # Build output paths
        output_pdf = validated_path.with_suffix(".pdf")
        output_html = validated_path.with_suffix(".html")

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

    return 0

if __name__ == "__main__":
    sys.exit(main())
