from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson
import pandas as pd
from weasyprint import HTML  # type: ignore[import-untyped]

from lib_analysis import HD, MLLI, TEST
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

    # Initialize results dictionary
    results: dict[str, dict[str, float]] = {}

    # Iterate over each JSON file
    for file in json_files:

        # Load JSON data
        with Path(file).open("r") as fin:
            data: dict[str, Any] = orjson.loads(fin.read())

        # Get metric config
        metric_config: dict[str, Any] = data["metric_config"]

        # Get queried data
        query_from_db: list[dict[str, Any]] = data["query_from_db"]

        # Filter database data
        filtered_db: pd.DataFrame = pd.DataFrame(query_from_db)

        # Store bootstrap cutoffs
        cutoffs: list[list[float, float]] = data["bootstrap"]["cutoffs"]

        # Process each group in the filtered data
        for group_label, group_data in filtered_db.groupby(["test", "recruitment_type", "recruitment_year", "gender"]):

            # Create a unique key for the group
            key = "_".join(str(x).lower() for x in group_label)

            # Extract values to standardize
            values = group_data["value"].to_numpy()

            # Apply standardization and compute step distribution
            standardized_stats = (
                apply_standardization(data_to_standardize=values, cutoffs=cutoffs)["standardized_step"]
                    .value_counts(normalize=True, sort=False)
                    .mul(100)
                    .round(1)
                    .reindex(range(1, len(metric_config["requested_percentiles"])+2), fill_value=0)
                    .sort_index()
                    .add_prefix("step")
            )

            # Add rif and store as dictionary
            results[key] = (
                pd.concat([
                    standardized_stats,
                    pd.Series({"rif": f"{metric_config['report']['header_letter']}3"}),
                ]).to_dict())

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
                "rif",
            ],
        )
        .sort_values(by=["test", "recruitment_type","gender","recruitment_year"])
    )

    # Save report data to CSV
    csv_output_path: Path = Path("./db/db_norms.csv")
    data.to_csv(csv_output_path, index=False)

    # Initialize tables list
    tables: list[Styler] = []

    # Iterate over tables (one for Hd, the other for Mlli)
    for table in (
        data.loc[data.recruitment_type == HD, :].drop(columns=["recruitment_type"]),
        data.loc[data.recruitment_type == MLLI, :].drop(columns=["recruitment_type"]),
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
                    .set_table_attributes('class="table-bordered full-width mb-xs"')
                    .relabel_index([(TEST[i[0]], i[1].upper()) for i in table_with_index.index], axis=0)
                    .relabel_index(["Concorso", "F1", "F2", "F3", "F4", "F5", "F6", "T"], axis=1)
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
