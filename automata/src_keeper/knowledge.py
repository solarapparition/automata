"""Background knowledge about the `src` package."""

from src.utilities.analyze_package import analyze_package


def format_summary(info_dict: dict, prefix: str = "") -> str:
    """Format a package summary."""
    summary_lines = []
    for name, info in info_dict["components"].items():
        summary_lines.append(f"{prefix}- {name}: {info['docstring_summary']}")
        if "components" in info:
            summary_lines.append(format_summary(info, prefix=prefix + "  "))
    return "\n".join(summary_lines)


def load() -> str:
    """Analyze the src package and return a summary."""
    package_info = analyze_package("src", top_level_only=True)
    return f'You have background knowledge regarding the `src` package. The information is as follows:\nsrc: {package_info["docstring_summary"]}\n{format_summary(package_info)}'


if __name__ == "__main__":
    print(load())
