"""CLI utility to generate a statistical analysis report from a data file.

This module validates an input file path supplied via the command line,
loads the JSON data, renders a Jinja2 HTML template, and converts it to
a PDF using WeasyPrint. It is intended to be used as an executable script.

Example:
    $ python report.py path/to/data.json
"""

import argparse
from pathlib import Path
import sys
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
import orjson
from weasyprint import HTML


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters:
    -----------
    None

    Returns:
    --------
    argparse.Namespace
        Parsed arguments with attribute:
        - filepath (str): Path to the data file to report.
    """
    parser = argparse.ArgumentParser(
        description="Generate statistical analysis report from data file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the data file to report",
    )

    return parser.parse_args()


def _validate_file_path(filepath: str) -> Path:
    """
    Validate that the file path exists, points to a regular file, and is non-empty.

    Parameters:
    -----------
    filepath : str
        Path to the file to validate.

    Returns:
    --------
    Path
        pathlib.Path object for the validated file.

    Raises:
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path is not a file or the file is empty.
    """
    file_path = Path("./data_out") / filepath / f"{filepath}_analysis.json"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    if not file_path.stat().st_size > 0:
        raise ValueError(f"File is empty: {filepath}")

    return file_path


def _load_data(filepath: Path) -> dict[str, Any]:
    """
    Load JSON data from file.

    Parameters:
    -----------
    filepath : Path
        Path to the JSON file containing report input data.

    Returns:
    --------
    dict
        Parsed JSON object as a dictionary.

    Raises:
    ------
    OSError
        If the file cannot be opened or read.
    orjson.JSONDecodeError
        If the file content is not valid JSON.
    """
    with filepath.open("rb") as fin:
        return orjson.loads(fin.read())


def main() -> int:
    """
    Entry point for report generation.

    Parses arguments, validates the input file path, loads JSON data, renders
    the 'report.html' Jinja2 template with that data, and writes a PDF using
    WeasyPrint to the same directory as the input file (with a .pdf extension).

    Returns:
    --------
    int
        Process exit status code:
        - 0: Success
        - 1: Invalid input path or validation error
        - 2: Rendering or PDF generation error
    """
    # Parse command line arguments
    args = _parse_arguments()

    # Validate the file path
    try:
        validated_path = _validate_file_path(args.filepath)
        print(f"Processing file: {validated_path}")
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
        data: dict[str, Any] = _load_data(validated_path)
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
