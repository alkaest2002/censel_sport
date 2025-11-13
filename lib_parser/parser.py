# mypy: disable-error-code="misc"

import argparse


def get_base_parser() -> argparse.ArgumentParser:
    """Create a base argument parser for data analysis.

    Returns:
        argparse.ArgumentParser: Configured argument parser with filepath argument.
    """
    parser = argparse.ArgumentParser(
        description="Generate statistical analysis and reporting from data file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--filepath", "-f",
        required=True,
        type=str,
        help="Path to the data file to analyze/report",
    )

    return parser


def get_base_report_parser() -> argparse.ArgumentParser:
    """Create a base argument parser for report generation.

    Returns:
        argparse.ArgumentParser: Configured argument parser with header letter
            and page number arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate db statistics report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--header-letter", "-l",
        required=True,
        type=str,
        help="Letter for report header section (e.g., 'A')",
    )

    parser.add_argument(
        "--page-number", "-n",
        required=True,
        type=int,
        help="Starting page number for report pages (e.g., 1)",
    )

    return parser


def get_report_parser() -> argparse.ArgumentParser:
    """Create a comprehensive argument parser for data analysis and reporting.

    Combines base parser functionality with report-specific options including
    header letter, page number, and recompute flag.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all required
            arguments for data analysis and report generation.
    """
    parser = get_base_parser()

    parser.add_argument(
        "--header-letter", "-l",
        required=True,
        type=str,
        help="Letter for report header section (e.g., 'A')",
    )

    parser.add_argument(
        "--page-number", "-n",
        required=True,
        type=int,
        help="Starting page number for report pages (e.g., 1)",
    )

    parser.add_argument(
        "--recompute", "-x",
        action="store_true",
        help="Re-run analysis",
    )

    return parser
