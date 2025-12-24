from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pandas as pd

from lib_analysis import HD, MLLI, TEST
from lib_analysis.utils_generic import query_db
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
    # Get parser
    parser: argparse.ArgumentParser = create_parser(page_number=True, header_letter=True)

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Retriev db
    db: pd.DataFrame = query_db({
        "recruitment_year": {
            "label": "Anni di reclutamento: 2022, 2023, 2024, 2025",
            "query": "recruitment_year.between(2022,2025)",
        },
    })

    # Define bin ages
    age_bins: list[int] = [13, 22, 29]

    # Bin age
    db["age_binned"] = pd.cut(
        db["age"], bins=age_bins,
        labels=["17-23", "24-29"],
        right=True,
    )

    # Group db by test and recruitment year
    hd_grouped: pd.DataFrame = (
        db.loc[db["recruitment_type"].eq(HD)]
            .groupby(["test", "recruitment_year"])
    )
    mlli_grouped: pd.DataFrame = (
        db.loc[db["recruitment_type"].eq(MLLI)]
        .groupby(["test", "recruitment_year"])
    )

    # Prepare data for the report
    tables_data: list[pd.DataFrame] = []

    # Iterate over HD and Mlli groups
    for grouped in (hd_grouped, mlli_grouped):

        # Initiliaze list to collect grouped stats
        grouped_stats: list[pd.DataFrame] = []

        # Iterate over each group
        for _, grouped_data in grouped:

            # Aggregate statistics
            grouped_agg: pd.DataFrame = (
                pd.concat([
                    grouped_data[["test", "recruitment_year", "gender"]]
                        .value_counts()
                        .reset_index(drop=False)
                        .pivot_table(index=["test", "recruitment_year"], columns="gender", observed=True),
                    grouped_data[["test", "recruitment_year", "age_binned"]]
                        .value_counts()
                        .reset_index(drop=False)
                        .pivot_table(index=["test", "recruitment_year"], columns="age_binned", observed=True),
                ], axis=1)
                .droplevel(level=0, axis=1)
                .reset_index(drop=False)
            )

            # Add total count by summing the last two columns
            grouped_agg = grouped_agg.assign(total_count=grouped_agg.iloc[:, -2:].sum(axis=1))

            # Append
            grouped_stats.append(grouped_agg)

        # Cache order of tests
        ordered_tests: list[str] = list(TEST.keys())

        # Create table from grouped stats
        table: pd.DataFrame = (
            pd.concat(grouped_stats, axis=0, ignore_index=True)
                # Sort by test according to predefined order
                .sort_values(
                    by=["test"], key=lambda x: x.map(lambda y, ordered_tests=ordered_tests: ordered_tests.index(y)),
                )
        )
        # Append table to data
        tables_data.append(table)

    try:

        # Render template with data
        _: dict[str, Path] = render_template(
            jinja_template_name="report_db_stats.html",
            output_folder=Path("./data_out/_report"),
            output_filename=f"{args.header_letter}_db_stats",
            output_formats=["pdf"],
            data=tables_data,
            header=args.header_letter,
            page=args.page_number,
        )

        # Print success message
        print("Database statistics generated successfully.")

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
