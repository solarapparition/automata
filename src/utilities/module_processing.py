"""Utilities for processing Python modules."""

import ast
from pathlib import Path
import re
from typing import List, Union


def extract_function_name(code_chunk: str) -> Union[str, None]:
    """Extract the name of a function from its definition text, if it exists."""
    pattern = r"def\s+(\w+)\s*\("
    match = re.search(pattern, code_chunk)
    if match:
        return match.group(1)
    return None


def construct_fn_id(
    function_name: str, module_prefix: List[str], class_name: Union[str, None]
) -> str:
    """Construct an ID for a function based on its name, module prefix, and class name."""
    id_parts = module_prefix.copy()
    if class_name:
        id_parts.append(class_name)
    id_parts.append(function_name)
    return ".".join(id_parts)


def construct_module_path(module_prefix: List[str], package_dir: Path) -> Path:
    """Construct the path to the module containing the function."""
    module_path = package_dir.joinpath(*module_prefix).with_suffix(".py")
    return module_path


def split_module_chunks(code: str) -> List[str]:
    """Split the code of a Python module.

    :param code: The code of the python module to split.
    :type code: str
    :return: A list of strings, each representing a chunk of the module code.
    :rtype: List[str]
    """
    module = ast.parse(code)
    module_docstring = ast.get_docstring(module)

    chunks = []

    if module_docstring:
        chunks.append(f'"""{module_docstring}"""')

    current_chunk = ""
    current_chunk_type = None

    for node in module.body:
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            if current_chunk_type != "import":
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                current_chunk = ""
            current_chunk += ast.unparse(node) + "\n"
            current_chunk_type = "import"
        elif isinstance(node, ast.Assign):
            if current_chunk_type != "variable":
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                current_chunk = ""
            current_chunk += ast.unparse(node) + "\n"
            current_chunk_type = "variable"
        elif isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ):
            if current_chunk:
                chunks.append(current_chunk.rstrip())
                current_chunk = ""
            current_chunk_type = None
            chunks.append(ast.unparse(node))
        elif isinstance(node, ast.ClassDef):
            if current_chunk:
                chunks.append(current_chunk.rstrip())
                current_chunk = ""
            current_chunk_type = None

            class_docstring = ast.get_docstring(node)
            if class_docstring:
                chunks.append(f'class {node.name}:\n    """{class_docstring}"""')
            else:
                chunks.append(f"class {node.name}:")

            for item in node.body:
                if isinstance(item, ast.Assign):
                    if current_chunk_type != "class_variable":
                        if current_chunk:
                            chunks.append(current_chunk.rstrip())
                        current_chunk = ""
                    current_chunk += "    " + ast.unparse(item) + "\n"
                    current_chunk_type = "class_variable"
                elif isinstance(item, ast.FunctionDef) or isinstance(
                    item, ast.AsyncFunctionDef
                ):
                    if current_chunk:
                        chunks.append(current_chunk.rstrip())
                        current_chunk = ""
                    current_chunk_type = None
                    chunks.append("    " + ast.unparse(item).replace("\n", "\n    "))

    if current_chunk:
        chunks.append(current_chunk.rstrip())

    return chunks


def demo() -> None:
    """Run a demo of the split_module_chunks function."""
    code = """
    import os
    import sys

    x = 10
    y = 20

    class MyClass:
        a = 5
        b = 10

        def my_method(self):
            pass

    import re
    from collections import defaultdict

    def my_function():
        pass

    z = 30
    """

    chunks = split_module_chunks(code)
    for chunk in chunks:
        print(chunk)


if __name__ == "__main__":
    demo()
