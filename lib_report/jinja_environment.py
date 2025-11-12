


from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from lib_analysis.utils_generic import format_seconds

# Define template dir
templates_dir = Path("./lib_report").resolve()


# Init jinja environment
jinja_env: Environment = Environment(
    # Use FileSystemLoader to load templates from the 'report' directory
    loader=FileSystemLoader(str(templates_dir)),
    # Enable strict undefined handling to catch missing variables
    undefined=StrictUndefined,
    # Auto-escape HTML for security
    autoescape=select_autoescape(["html", "xml"]),
)

jinja_env.filters["format_seconds"] = format_seconds
