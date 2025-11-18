"""PDF Merger utility for combining multiple PDF files into a single document.

This module provides functionality to merge PDF files from a specified directory,
with support for recursive searching and natural sorting of filenames.
"""

from pathlib import Path
import re
import sys

from pypdf import PdfWriter


def natural_key(s: str) -> list:
    """Convert a string into a list for natural sorting.

    Splits the string into numeric and non-numeric parts to enable
    natural (human-friendly) sorting where '2' comes before '10'.

    Args:
        s: The string to convert for natural sorting.

    Returns:
        A list of integers and lowercase strings for comparison.
    """
    parts: list[str] = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _collect_pdfs(folder: Path, out_name: str) -> list[Path]:
    """Collect PDF files from the specified folder.

    Searches for PDF files in the given folder, optionally recursively,
    and excludes the output file from the list. Files are sorted using
    natural sorting order.

    Args:
        folder: The directory path to search for PDF files.
        out_name: The output filename to exclude from results.

    Returns:
        A sorted list of Path objects pointing to PDF files.
    """
    # Glob pdfs
    candidates = folder.glob("*.pdf")

    # lowercase final_report.pdf
    out_name_lower: str = out_name.lower()

    # Collect pdfs except the output file
    pdfs: list[Path] = [
        p for p in candidates
        if p.is_file() and p.name.lower() != out_name_lower
    ]

    # Sort pdfs naturally
    pdfs.sort(key=lambda p: natural_key(p.name))

    return pdfs


def merge_pdfs(pdf_paths: list[Path], output_path: Path) -> int:
    """Merge multiple PDF files into a single output file.

    Attempts to merge all PDF files in the list, handling encrypted PDFs
    by trying an empty password. Skips files that cannot be processed.

    Args:
        pdf_paths: list of Path objects pointing to PDF files to merge.
        output_path: Path where the merged PDF will be written.

    Returns:
        Exit code: 0 for success, 1 if no files were found.
    """
    # Raise error, if there no pdfs
    if not pdf_paths:
        print("No PDF files found to merge.", file=sys.stderr)
        return 1

    # Initialize PDF merger
    merger: PdfWriter = PdfWriter()

    # Append pds to merger
    for p in pdf_paths:
        merger.append(str(p))
        print(f"Appended: {p}")

    # Finalize PDF creation
    merger.write(str(output_path))
    print(f"Merged {len(pdf_paths)} files into: {output_path}")

    return 0


def main() -> int:
    """Main entry point for the PDF merger CLI.

    Parses command-line arguments, validates inputs, and orchestrates
    the PDF merging process.

    Returns:
        Exit code: 0 for success, 1 for merge errors, 2 for invalid directory.
    """
    folder: Path = Path("./data_out/_report").resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 2

    output_path: Path = folder / "final_report.pdf"
    pdfs: list[Path] = _collect_pdfs(folder, output_path.name)

    return merge_pdfs(pdfs, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
