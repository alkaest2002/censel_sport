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

from lib_analysis.utils_generic import load_configuration_data, parse_arguments, validate_file_path


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
    args = parse_arguments()

    # Validate the file path
    try:
        validated_path = validate_file_path(args.filepath, "report")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    # Init jinja environment
    templates_dir = Path("./report").resolve()
    jinja_env: Environment = Environment(
        # Use FileSystemLoader to load templates from the 'report' directory
        loader=FileSystemLoader(str(templates_dir)),
        # Enable strict undefined handling to catch missing variables
        undefined=StrictUndefined,
        # Auto-escape HTML for security
        autoescape=select_autoescape(["html", "xml"]),
    )

    # Load data and render template, then generate PDF
    try:
        data: dict[str, Any] = load_configuration_data(validated_path)
        template = jinja_env.get_template("report.html")
        rendered_html: str = template.render(data=data)
        output_pdf = validated_path.with_suffix(".pdf")
        output_html = validated_path.with_suffix(".html")

        # Write html
        with output_html.open("w") as fout:
            fout.write(rendered_html)

        # Write PDF
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))
        print(f"Report generated: {output_pdf}")
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
