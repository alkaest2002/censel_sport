from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from lib_report.utils_jinja import (
    format_number_locale,
    format_seconds,
    format_title,
    get_recruitment_type_label,
    get_test_label,
)


def _get_templates_directory() -> Path:
    """Get the resolved path to the templates directory.

    Returns:
        Path: Resolved path to the lib_report templates directory.
    """
    return Path("./lib_report").resolve()


def _create_jinja_environment(templates_dir: Path) -> Environment:
    """Create and configure a Jinja2 environment.

    Args:
        templates_dir: Path to the directory containing Jinja2 templates.

    Returns:
        Environment: Configured Jinja2 environment with security features enabled.
    """
    return Environment(
        # Use FileSystemLoader to load templates from the 'report' directory
        loader=FileSystemLoader(str(templates_dir)),
        # Enable strict undefined handling to catch missing variables
        undefined=StrictUndefined,
        # Auto-escape HTML for security
        autoescape=select_autoescape(["html", "xml"]),
    )


# Define template directory
templates_dir = _get_templates_directory()

# Initialize Jinja2 environment with security configurations
jinja_env: Environment = _create_jinja_environment(templates_dir)

# Register custom filters
jinja_env.filters["format_seconds"] = format_seconds
jinja_env.filters["format_title"] = format_title
jinja_env.filters["test_label"] = get_test_label
jinja_env.filters["recruitment_type_label"] = get_recruitment_type_label
jinja_env.filters["format_number_locale"] = format_number_locale
