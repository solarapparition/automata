"""This module contains functionality to analyze a package and extract docstrings."""

import ast
import os
from pprint import pprint
from typing import Any, Dict, Union


def get_docstring(obj, summary: bool = True) -> str:
    """Get the docstring for an object."""
    docstring = ast.get_docstring(obj)
    if docstring and summary:
        return docstring.split(".")[0]
    if docstring:
        return docstring
    return ""


def extract_member_variables(item: Union[ast.Assign, ast.AnnAssign]) -> Dict[str, Any]:
    """Extract member variables from an object."""
    member_variables = {}
    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
    for target in targets:
        if isinstance(target, ast.Name):
            variable_name = target.id
            variable_docstring = ""
            if isinstance(item, ast.Assign) and isinstance(item.value, ast.Str):
                variable_docstring = item.value.s
            member_variables[variable_name] = {
                "docstring_summary": variable_docstring,
                "components": {},
            }
    return member_variables


PackageInfo = Dict[str, Union[str, Dict[str, "PackageInfo"]]]


def extract_info(node: ast.AST, top_level_only: bool, visited=None) -> PackageInfo:
    """Extract info from a node."""
    if visited is None:
        visited = set()

    if id(node) in visited:
        return {}

    visited.add(id(node))

    info = {"docstring_summary": get_docstring(node, summary=True)}

    if not top_level_only and isinstance(node, (ast.Module, ast.ClassDef)):
        components = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                components[item.name] = extract_info(item, top_level_only, visited)
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                components.update(extract_member_variables(item))
        info["components"] = components
    return info


def is_init_py(file: str) -> bool:
    """Check if a file is an __init__.py file."""
    return file == "__init__.py"


def is_package_dir(root: str) -> bool:
    """Check if a directory is a package directory."""
    return "__init__.py" in os.listdir(root)


def get_relative_module_name(package_path: str, file_path: str) -> str:
    """Get the relative module name from a file path."""
    relative_path = os.path.relpath(file_path, package_path)
    return relative_path.replace(os.path.sep, ".").strip(".py")


def analyze_module(module_path: str, top_level_only: bool = False) -> Dict[str, Any]:
    """Analyze a single module and return its info."""
    with open(module_path, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read())
        return extract_info(tree, top_level_only)


def analyze_package(package_path: str, top_level_only: bool = False) -> PackageInfo:
    """Analyze a package and return its info."""
    root_path = package_path

    def analyze_package_recursive(
        package_path: str, top_level_only: bool = False, root_path: str = None
    ) -> PackageInfo:
        package_info = {"docstring_summary": "", "components": {}}

        for entry in os.scandir(package_path):
            if entry.is_file() and entry.name.endswith(".py"):
                file_path = entry.path
                module_name = get_relative_module_name(root_path, file_path)
                module_info = analyze_module(file_path, top_level_only)
                if is_init_py(entry.name):
                    package_info["docstring_summary"] = module_info["docstring_summary"]
                else:
                    package_info["components"][module_name] = module_info
            elif entry.is_dir() and is_package_dir(entry.path) and top_level_only:
                package_name = get_relative_module_name(root_path, entry.path)
                package_info["components"][package_name] = analyze_package_recursive(
                    entry.path, top_level_only, root_path
                )

        return package_info

    return analyze_package_recursive(package_path, top_level_only, root_path)


def flatten_package_dict(package_info: Dict) -> Dict[str, str]:
    """Flatten a package dict into a single dict with fully qualified names as keys."""

    def flatten_info(info: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        flat_dict = {}
        if prefix:
            prefix += "."

        if "docstring_summary" in info:
            flat_dict[prefix[:-1]] = info["docstring_summary"]

        if "components" in info:
            for component_name, component_info in info["components"].items():
                flat_dict.update(flatten_info(component_info, prefix + component_name))

        return flat_dict

    flat_info = {}

    for module_name, module_info in package_info.items():
        flat_info.update(flatten_info(module_info, module_name))
    return flat_info


def test_analyze_package_top_level_only():
    """Test analyze_package with top_level_only=True."""
    package_path = "test_data/test_package"  # Path to the test package
    result = analyze_package(package_path, top_level_only=True)

    # Check if the main package docstring summary is present
    assert "docstring_summary" in result

    # Check if module_a and module_b have only the docstring summary
    for module_name in ["module_a", "module_b"]:
        assert "docstring_summary" in result[module_name]
        assert "components" not in result[module_name]

    # Check if the sub-package docstring summary is present
    assert "sub_package" in result["components"]
    assert "docstring_summary" in result["components"]["sub_package"]

    pprint(result)


if __name__ == "__main__":
    test_analyze_package_top_level_only()
