# mypy: disable-error-code="misc"

import argparse


def get_base_parser() -> argparse.ArgumentParser:
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
        description="Generate statistical analysis and reporting from data file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the data file to analyze/report",
    )

    return parser
