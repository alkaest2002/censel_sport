from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson
import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_analysis import MT100, MT1000, PUSHUPS, SITUPS, SWIM25
from lib_analysis.utils_generic import query_from_db
from lib_analysis.utils_stats import apply_standardization
from lib_parser.parser import create_parser
from lib_report.jinja_environment import jinja_env, templates_dir

if TYPE_CHECKING:
    import argparse

    import jinja2
    from pd.io.formats.style import Styler


def main() -> int:
    """Generate database statistics report.

    Returns:
        int: Process exit status code:
            - 0: Success
            - 1: Rendering or PDF generation error
    """
    # Get parser
    parser: argparse.ArgumentParser = create_parser(header_letter=True, page_number=True)

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Globally find all analysis JSON files
    json_files = list(Path("./data_out").glob("**/*_analysis.json"))

    # Load db
    db: pd.DataFrame = pd.read_csv(Path("./db") / "db.csv")

    # Initialize results dictionary
    results = {}

    # Iterate over each JSON file
    for file in json_files:

        # Load JSON data
        with Path(file).open("r") as fin:
            data: dict[str, Any] = orjson.loads(fin.read())

        # Get metric config
        metric_config: dict[str, Any] = data["metric_config"]

        # Get stratification
        stratification: dict[str, Any] = metric_config["stratification"]

        # Filter database data
        filtered_db: pd.DataFrame = query_from_db(stratification, db)

        # Store bootstrap cutoffs
        cutoffs: list[list[float, float]] = data["bootstrap"]["cutoffs"]

        # Process each group in the filtered database
        for group_label, group_data in filtered_db.groupby(["test", "recruitment_type", "recruitment_year", "gender"]):

            # Create a unique key for the group
            key = "_".join(str(x).lower() for x in group_label)

            # Extract values to standardize
            values = group_data["value"].to_numpy()

            # Apply standardization and compute step distribution
            results[key] = (
                apply_standardization(data_to_standardize=values, cutoffs=cutoffs)["standardized_step"]
                    .value_counts(normalize=True, sort=False)
                    .mul(100)
                    .round(1)
                    .sort_index()
                    .add_prefix("step")
                    .to_dict()
            )

            # Add total count
            results[key]["total_count"] = len(values)

    # Convert results to DataFrame
    results_df: pd.DataFrame = pd.DataFrame(results).fillna(0).T

    # Prepare report data
    data: pd.DataFrame = (
        pd.DataFrame(
            np.hstack(
                [
                    pd.Series(results_df.index).str.rsplit("_", n=3, expand=True).to_numpy(),
                    results_df.to_numpy(),
                ],
            ),
            columns=[
                "test",
                "recruitment_type",
                "recruitment_year",
                "gender",
                "step1",
                "step2",
                "step3",
                "step4",
                "step5",
                "step6",
                "total_count"],
        )
        .sort_values(by=["test", "recruitment_type","gender","recruitment_year"])
    )

    # Save report data to CSV
    csv_output_path: Path = Path("./db/db_norms.csv")
    data.to_csv(csv_output_path, index=False)

    # Initialize tables list
    tables: list[Styler] = []

    # Create dictionary of tests
    tests_labels: dict[str, str] = ({
        MT100: "Corsa 100 mt",
        MT1000: "Corsa 1000 mt",
        SWIM25: "Nuoto 25 mt",
        SITUPS: "Addominali",
        PUSHUPS: "Piegamenti sulle braccia",
    })

    # Iterate over tables (one for hd, the other for mlli)
    for table in (
        data.loc[data.recruitment_type == "hd", :],
        data.loc[data.recruitment_type == "mlli", :],
    ):
        # Create indexed table
        table_with_index: pd.DataFrame = (
            table
                .rename(columns={"test": "Prova sportiva", "gender": "Genere" })
                .set_index(["Prova sportiva", "Genere"])
        )

        # Style table via Pandas (and not Jinja, due to complex layout)
        table_styler: Styler = (
            table_with_index
                .style
                    .format(precision=1)
                    .hide(subset=["recruitment_type"], axis=1)
                    .set_table_attributes('class="table-bordered full-width mb-xs"')
                    .relabel_index([(tests_labels[i[0]], i[1].upper()) for i in table_with_index.index], axis=0)
                    .relabel_index(["Concorso", "F1", "F2", "F3", "F4", "F5", "F6", "Tot"], axis=1)
        )

        # Add global styles
        table_styler.set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                ]},
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("border-bottom", ".5px solid #333"),
                ],
            },
        ])

        # Append styler
        tables.append(table_styler)

    # Store HTML tables for reporting
    tables_data: list[str] = [ t.to_html() for t in tables ]

    try:

        # Get db report template
        template: jinja2.Template = jinja_env.get_template("report_db_norms.html")

        # Build output path
        base_path: Path = Path(f"./data_out/_report/{args.header_letter}_db_norms")
        output_pdf: Path = base_path.with_suffix(".pdf")

        # Render template with data
        rendered_html: str =\
            template.render(tables_data=tables_data, header=args.header_letter, page=args.page_number)

        # Write PDF file
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf))
        print(f"Report generated: {output_pdf}")

    # Handle exceptions
    except Exception as e:  # noqa: BLE001
        print(f"Error while generating report: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
