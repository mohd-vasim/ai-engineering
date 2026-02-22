"""
Converts a Jupyter Notebook (.ipynb) file OR all notebooks in a folder
to Markdown files and stores outputs in a separate export folder.
"""

import os
import subprocess
import shutil
import argparse


def convert_single_notebook(notebook_path, root_dir, export_dir):
    """Convert one notebook to markdown preserving relative path."""
    dirpath = os.path.dirname(notebook_path)
    rel_path = os.path.relpath(dirpath, root_dir) if root_dir else ""

    output_subdir = os.path.join(export_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(notebook_path))[0]
    md_file = os.path.join(output_subdir, f"{base_name}.md")
    files_folder = os.path.join(output_subdir, f"{base_name}_files")

    # Remove old outputs
    if os.path.exists(md_file):
        os.remove(md_file)
        print(f"🗑️ Removed old: {md_file}")
    if os.path.isdir(files_folder):
        shutil.rmtree(files_folder)
        print(f"🗑️ Removed old folder: {files_folder}")

    # Convert
    print(f"🔄 Converting: {notebook_path}")
    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "markdown",
                "--output",
                base_name,
                "--output-dir",
                output_subdir,
                notebook_path,
            ],
            check=True,
        )
        print(f"✅ Done: {md_file}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {notebook_path}: {e}")


def convert_notebooks(input_path, export_dir):
    """Convert a single notebook OR all notebooks in a folder."""
    input_path = os.path.abspath(input_path)
    export_dir = os.path.abspath(export_dir)

    os.makedirs(export_dir, exist_ok=True)

    if os.path.isfile(input_path) and input_path.endswith(".ipynb"):
        # Single file mode
        root_dir = os.path.dirname(input_path)
        convert_single_notebook(input_path, root_dir, export_dir)

    elif os.path.isdir(input_path):
        # Folder mode
        root_dir = input_path
        for dirpath, _, filenames in os.walk(input_path):
            for filename in filenames:
                if filename.endswith(".ipynb") and not filename.startswith("."):
                    notebook_path = os.path.join(dirpath, filename)
                    convert_single_notebook(notebook_path, root_dir, export_dir)
    else:
        print(f"❌ Invalid input: {input_path} (must be .ipynb file or folder)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Jupyter Notebook file OR a folder of notebooks "
        "to Markdown and store in an export folder."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a .ipynb file OR a folder containing notebooks",
    )
    parser.add_argument(
        "--export",
        type=str,
        default="markdown-version",
        help="Folder where markdown files should be saved",
    )

    args = parser.parse_args()
    convert_notebooks(args.input, args.export)