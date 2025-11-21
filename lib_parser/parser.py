# mypy: disable-error-code="misc"

import argparse


def create_parser(
    filepath: bool = False,
    page_number: bool = False,
    header_letter: bool = False,
    recompute: bool = False,
) -> argparse.ArgumentParser:
    """Create a customizable argument parser with specified parameters.

    Args:
        filepath: Include --filepath argument.
        page_number: Include --page-number argument.
        header_letter: Include --header-letter argument.
        recompute: Include --recompute argument.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if filepath:
        parser.add_argument(
            "--filepath", "-f",
            required=True,
            type=str,
            help="Path to the data file",
        )

    if page_number:
        parser.add_argument(
            "--page-number", "-n",
            required=True,
            type=int,
            help="Starting page number for report pages (e.g., 1)",
        )

    if header_letter:
        parser.add_argument(
            "--header-letter", "-l",
            required=True,
            type=str,
            help="Letter for report header section (e.g., 'A')",
        )

    if recompute:
        parser.add_argument(
            "--recompute", "-x",
            action="store_true",
            help="Force recomputation of cached results",
        )

    return parser
