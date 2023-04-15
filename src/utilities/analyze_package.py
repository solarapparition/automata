"""This module contains functionality to analyze a package and extract docstrings."""

import os
import ast
from typing import Any, Dict, Union


def get_docstring(obj, summary: bool=True) -> str:
    """Get the docstring for an object."""
    docstring = ast.get_docstring(obj)
    if docstring and summary:
        return docstring.split(".")[0]
    if docstring:
        return docstring
    return ""


def extract_member_variables(item: Union[ast.Assign, ast.AnnAssign]) -> Dict[str, Any]:
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


def extract_info(node: ast.AST, visited=None) -> Dict[str, Any]:
    """Extract info from a node."""
    if visited is None:
        visited = set()

    if id(node) in visited:
        return {}

    visited.add(id(node))

    info = {"docstring_summary": get_docstring(node, summary=True)}

    # try:
    #     if node.name == "TargetIngestion":
    #         breakpoint()
    # except AttributeError:
    #     pass
    # AnnAssign

    if isinstance(node, (ast.Module, ast.ClassDef)):
        components = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                components[item.name] = extract_info(item, visited)
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                components.update(extract_member_variables(item))
        info["components"] = components
    return info


def analyze_module(module_path: str) -> Dict[str, Any]:
    """Analyze a single module and return its info."""
    with open(module_path, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read())
        return extract_info(tree)

def analyze_package(package_path: str) -> Dict[str, Dict[str, Any]]:
    package_dict = {}

    for root, _, files in os.walk(package_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                module_name = file_path.replace("/", ".").strip(".py")
                package_dict[module_name] = analyze_module(file_path)

                # with open(file_path, "r") as source:
                #     tree = ast.parse(source.read())
                #     package_dict[module_name] = extract_info(tree)

    return package_dict



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
