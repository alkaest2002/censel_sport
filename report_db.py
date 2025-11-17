from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_parser.parser import get_db_report_parser
from lib_report.jinja_environment import jinja_env, templates_dir

if TYPE_CHECKING:
    import argparse

    import jinja2


def _recruitment_year_fn(
        test: str,  # noqa: ARG001
        gender: str,  # noqa: ARG001
        age_binned: str,  # noqa: ARG001
        df: pd.DataFrame,
    ) -> str:
    """Get recruitment years for a given group.

    Args:
        x: DataFrame group containing data
        test: Test type
        gender: Gender
        age_binned: Age bin
        df: Entire DataFrame

    Returns:
        str: String representation of recruitment years count dictionary.
    """
    query_string ="test==@test and gender==@gender and age_binned==@age_binned"
    years = df.query(query_string).loc[:,"recruitment_year"].value_counts()
    return str(years.to_dict())[1:-1]

def _gender_fn(
        test: str,  # noqa: ARG001
        recruitment_year: int,  # noqa: ARG001
        age_binned: str,  # noqa: ARG001
        df: pd.DataFrame,
    ) -> str:
    """Get recruitment years for a given group.

    Args:
        x: DataFrame group containing data
        test: Test type
        recruitment_year: Year
        age_binned: Age bin
        df: Entire DataFrame

    Returns:
        str: String representation of recruitment years count dictionary.
    """
    query_string ="test==@test and recruitment_year==@recruitment_year and age_binned==@age_binned"
    genders = df.query(query_string).loc[:,"gender"].value_counts()
    return str(genders.to_dict())[1:-1]

def main() -> int:
    """Generate database statistics report.

    This function reads a CSV database file, processes the data to compute
    statistics including duplicates percentage and age-binned summaries,
    then generates both HTML and PDF reports using Jinja templates.

    Returns:
        int: Process exit status code:
            - 0: Success
            - 1: Error in file validation or loading
            - 2: Rendering or PDF generation error
    """
    # Get report parser
    parser: argparse.ArgumentParser = get_db_report_parser()

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Define file path
    filepath: Path = Path("db/db.csv")

    # Read CSV file into DataFrame
    df: pd.DataFrame = pd.read_csv(filepath)

    # Compute percentage of duplicates
    duplicated: float = round((df.duplicated().sum() / df.shape[0]) * (100), 2)

    # Define bin ages
    age_bins: list[int] = [13, 29, 39, 49, 59, 69, 79]

    # Bin age
    df["age_binned"] = pd.cut(
        df["age"], bins=age_bins,
        labels=["14-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
        right=True,
    ).astype(str)

    # Compute summary table
    gender_data: list[dict[str, Any]] = []
    years_data: list[dict[str, Any]] = []

    for _, g_data in df.groupby(["test", "recruitment_year", "age_binned"], observed=True):
        test = g_data.iloc[0]["test"]
        recruitment_year = g_data.iloc[0]["recruitment_year"]
        age_binned = g_data.iloc[0]["age_binned"]
        current_group_data = ({
            "test": test,
            "recruitment_year":recruitment_year,
            "age_binned":age_binned,
            "gender": _gender_fn(test, recruitment_year, age_binned, df),
            "counts": g_data.shape[0],
        })
        gender_data.append(current_group_data)

    for _, g_data in df.groupby(["test", "gender", "age_binned"], observed=True):
        test = g_data.iloc[0]["test"]
        gender = g_data.iloc[0]["gender"]
        age_binned = g_data.iloc[0]["age_binned"]
        current_group_data = ({
            "test": test,
            "gender":gender,
            "age_binned":age_binned,
            "recruitment_year":_recruitment_year_fn(test, gender, age_binned, df),
            "counts": g_data.shape[0],
        })
        years_data.append(current_group_data)

    # Load data
    try:
        # Get report template
        template: jinja2.Template = jinja_env.get_template("db_stats.html")

        # Build output paths
        base_path: Path = Path("./data_out/db_stats")
        output_pdf: Path = base_path.with_suffix(".pdf")
        output_html: Path = base_path.with_suffix(".html")

        # Render HTML
        rendered_html: str =\
            template.render(data={
                "gender_data": gender_data,
                "years_data": years_data,
            }, header=args.header_letter, page=args.page_number)

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

    print(f"Data loaded and validated successfully. Percentage of duplicates {duplicated}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
