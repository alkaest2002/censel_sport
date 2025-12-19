from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import orjson
from weasyprint import HTML

from lib_report.jinja_environment import jinja_env, templates_dir

OutputFormat = Literal["html", "pdf", "json"]

if TYPE_CHECKING:
    import jinja2


def render_template(
    jinja_template_name: str,
    output_folder: Path,
    output_filename: str,
    output_formats: list[OutputFormat] | OutputFormat,
    data: dict[str, Any],
    **template_kwargs: Any,
) -> dict[OutputFormat, Path]:
    """
    Render a Jinja template and/or export data to multiple formats.

    Args:
        jinja_template_name: The name of the Jinja2 template to render (required for html/pdf formats)
        output_folder: Folder where the output files will be saved
        output_filename: Base filename for the output (without extension)
        output_formats: Output format(s) - can be single format or list of formats
               - "html": Render template to HTML
               - "pdf": Render template to PDF
               - "json": Export data as JSON
        data: Dictionary containing the data to pass to the template or export
        **template_kwargs: Additional keyword arguments to pass to the template

    Returns:
        Dictionary mapping format to output file path
    """
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Convert single format to list
    if isinstance(output_formats, str):
        output_formats = [output_formats]

    # Initialize output paths dictionary
    output_paths: dict[OutputFormat, Path] = {}

    # Prepare template context
    template_context = {"data": data, **template_kwargs }

    # Get Jinja template
    jinja_template: jinja2.Template = jinja_env.get_template(jinja_template_name)

    # Render template
    rendered_html = jinja_template.render(**template_context)

    # Write HTML if requested
    if "html" in output_formats:
        output_html_path = (output_folder / output_filename).with_suffix(".html")
        with output_html_path.open("w", encoding="utf-8") as fout:
            fout.write(rendered_html)
        output_paths["html"] = output_html_path

    # Write PDF if requested
    if "pdf" in output_formats:
        output_pdf_path = (output_folder / output_filename).with_suffix(".pdf")
        HTML(string=rendered_html, base_url=str(templates_dir)).write_pdf(str(output_pdf_path))
        output_paths["pdf"] = output_pdf_path

    # Handle data export formats
    if "json" in output_formats:
        output_json_path = (output_folder / output_filename).with_suffix(".json")
        with output_json_path.open("wb") as fout:
            fout.write(orjson.dumps(data))
        output_paths["json"] = output_json_path

    return output_paths
