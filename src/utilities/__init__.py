"""
Utilities for the package.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import functools
import hashlib
from importlib import import_module
from itertools import islice, chain
import json
import os
from pathlib import Path
import shutil
import time
from types import ModuleType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd
import requests


def add_benchmarking(func: Callable, outdir: str, printout: bool = True) -> Callable:
    """Decorator to benchmark a function."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        outpath = os.path.join(
            outdir, f"benchmark_results_{generate_timestamp_id()}.txt"
        )
        benchmark_start = time.perf_counter()
        result = await func(*args, **kwargs)
        benchmarked_time = time.perf_counter() - benchmark_start
        with open(outpath, "w", encoding="utf-8") as file:
            print("Total Benchmarekd Time:", benchmarked_time, file=file)
        if printout:
            print("Benchmarking output written to:", Path(outpath).absolute())
        return result

    wrapper.__annotations__ = func.__annotations__
    return wrapper


def batched(iterable: Iterable, n: int) -> Iterable:
    """Batch data into tuples of size n. The last batch may be smaller."""
    if n < 1:
        raise ValueError("n must be >= 1")
    while batch := tuple(islice(iter(iterable), n)):
        yield batch


def convert_to_excel(jsons_dir: str, outpath: str):
    """Convert a directory of JSON files to an Excel file."""
    with pd.ExcelWriter(Path(outpath)) as writer:
        for filename in os.listdir(jsons_dir):
            with open(Path(jsons_dir) / filename, "r", encoding="utf-8") as file:
                table_data = json.load(file)
            table_name = filename.split(".", maxsplit=1)[0]
            table_data = pd.DataFrame.from_dict(table_data)
            table_data.to_excel(writer, sheet_name=table_name, index=False)


def get_key_value(vals: "Union[Mapping, Sequence]", key_path: tuple) -> Any:
    """
    Get a value from a nested dictionary or list using a sequence of keys.

    Parameters
    ----------
    vals : Union[Mapping, Sequence]
        The dictionary or list to search.
    key_path : tuple
        The path to the value to get.

    Returns
    -------
    Any
        The value at the specified path.

    Examples
    --------
    >>> get_key_value({"a": {"b": 1}, "c": [0, 1, 2]}, ("a", "b"))
    1
    >>> get_key_value({"a": {"b": 1}, "c": [0, 1, 2]}, ("c", 2))
    2
    >>> get_key_value({"a": {"b": 1}, "c": [0, 1, 2]}, ("d",))
    None
    """
    if not key_path:
        return vals
    try:
        return get_key_value(vals[key_path[0]], key_path[1:])
    except (KeyError, TypeError):
        return None


def extract_field(dict_sequence: "Iterable[Mapping]", path: tuple) -> Iterable:
    """
    Extract a field from a sequence of mappings, such as a list of dictionaries.

    Parameters
    ----------
    dict_sequence : Iterable[Mapping]
        The sequence of mappings to extract from.
    path : tuple
        The path to the field to extract.

    Returns
    -------
    Iterable
        The extracted field.

    Examples
    --------
    >>> extract_field(
        [
            {
                "a": {"b": 1},
                "c": [0, 1, 2]
            },
            {
                "a": {"b": 3},
                "c": [4, 5, 6]
            }
        ],
        ("a", "b"))
    [1, 3]
    """
    return (get_key_value(entry, path) for entry in dict_sequence)


def flatten_dict(data: "Union[dict, list]") -> dict:
    """
    Flatten a nested dictionary or list.

    Parameters
    ----------
    data : Union[dict, list]
        The dictionary or list to flatten.

    Returns
    -------
    dict
        The flattened dictionary.

    Examples
    --------
    >>> flatten_dict(
        {
            "a": {
                "b": 1,
                "c": 2
            },
            "d": [0, 1, 2]
        })
    {
        ("a", "b"): 1,
        ("a", "c"): 2,
        ("d", 0): 0,
        ("d", 1): 1,
        ("d", 2): 2
    }
    """
    result = {}
    items = ()
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = enumerate(data)
    for key, val in items:
        if isinstance(val, (dict, list)):
            result.update(
                {(key, *subk): subv for subk, subv in flatten_dict(val).items()}
            )
        else:
            result[(key,)] = val
    return result


async def gather_in_batches(*awaitables: Awaitable, batch_size: int) -> list:
    """
    Gather and wait for awaitables in batches.
    Useful for limiting the number of concurrent tasks requiring a resource that has a
    limited number of connections, such as a database.

    Parameters
    ----------
    *awaitables : Awaitable
        The awaitables to gather.
    batch_size : int
        The number of awaitables to gather at a time.
        Each batch will be awaited before the next batch is gathered.

    Returns
    -------
    list
        The results of the awaitables.

    Examples
    --------
    >>> await gather_in_batches(*[asyncio.sleep(1) for _ in range(10)], batch_size=3)
    [None, None, None, None, None, None, None, None, None, None]
    # gathers in batches of 3, so takes 4 seconds to execute
    """
    if len(awaitables) <= batch_size:
        return await asyncio.gather(*awaitables)
    return (
        await gather_in_batches(*awaitables[:batch_size], batch_size=batch_size)
    ) + (await gather_in_batches(*awaitables[batch_size:], batch_size=batch_size))


def generate_timestamp_id() -> str:
    """Generate an id based on the current timestamp."""
    return datetime.utcnow().strftime("%Y-%m-%d_%H%M-%S-%f")


def load_json_lists(directory: str) -> Iterable[Mapping]:
    """Load JSON data from individual files. Each file is assumed to be a list of JSON objects."""

    def load_json(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    return chain.from_iterable(
        load_json(Path(directory) / file)
        for file in os.listdir(directory)
        if file.endswith(".json")
    )


def package_func_args(
    func: Callable,
    args: tuple,
    kwargs: dict,
    id: str = "",  # pylint: disable=redefined-builtin, invalid-name
) -> "tuple[str, str, tuple, tuple[tuple[str, Any]]]":
    """
    Package function arguments for hashing.
    Use `id` to distinguish between different functions with the same `__name__` attribute.
    """
    return (id, func.__name__, args, tuple(kwargs.items()))


def quick_import(location: Path) -> ModuleType:
    """Import a module directly from a Path."""
    return import_module(str(location).replace(os.path.sep, "."))


def sample_with_handling(
    df: pd.DataFrame, sample_size: "Union[int, None]" = None, random_state: int = 0
) -> pd.DataFrame:
    """Sample a dataframe, handling the case where the sample size is larger than the dataframe."""
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(sample_size, random_state=random_state)


def search_nested_records(
    records: Iterable[Mapping[str, Union[str, Sequence[Mapping[str, str]]]]],
    values_to_search: Iterable[Mapping[str, str]],
    extractor: Callable[[Mapping, Mapping], str],
    nested_record_key: str,
    lookup_keys: Tuple[str],
) -> Iterable:
    """
    Efficiently search records containing nested values and extract record ids.

    Parameters
    ----------
    records : Iterable[Mapping[str, Union[str, Sequence[Mapping[str, str]]]]]
        The records to search.
    values_to_search : Iterable[Mapping[str, str]]
        The values to search for.
    extractor : Callable[[Mapping, Mapping], str]
        A function that extracts the record id from the record and nested value.
    nested_record_key : str
        The key in the record that contains the nested values.
    lookup_keys : Tuple[str]
        The keys in the nested values to use for searching.

    """

    # Create a dictionary to store record ids based on their nested values
    record_lookup: Dict[Tuple, List] = {}

    # Iterate through the records and populate the lookup dictionary
    for record in records:
        for record_data_dict in record[nested_record_key]:
            filtered_dict = {
                key: value
                for key, value in record_data_dict.items()
                if key in lookup_keys
            }
            record_data_tuple = tuple(sorted(filtered_dict.items()))
            if record_data_tuple not in record_lookup:
                record_lookup[record_data_tuple] = []
            extracted_value = extractor(record, record_data_dict)
            if extracted_value not in record_lookup[record_data_tuple]:
                record_lookup[record_data_tuple].append(extracted_value)

    # Search for the `values_to_search` in the lookup dictionary and store the results
    result = []
    for search_value in values_to_search:
        search_value_tuple = tuple(sorted(search_value.items()))
        extracted_values = record_lookup.get(search_value_tuple)
        if extracted_values:
            result.append(extracted_values[0])
        else:
            result.append(None)
    return result


def send_teams_notification(
    text: str, title: str, link: str, link_text: str, webhook_url: str
) -> int:
    """Send a notification to Microsoft Teams."""
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "title": title,
        "text": text,
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": link_text,
                "targets": [{"os": "default", "uri": link}],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(webhook_url, json=json.dumps(payload), headers=headers)
    return response.status_code


def stable_hash(value: Any, json_default: Union[Callable, None] = None) -> str:
    """
    Generate a stable hash for a value that does not change between different Python runs.
    The value must be JSON serializable.
    """
    json_str = json.dumps(
        value,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    )
    hash_str = hashlib.sha224(json_str.encode("utf-8")).hexdigest()
    return hash_str


def to_async(
    func: Callable,
    pool: "Union[ThreadPoolExecutor, ProcessPoolExecutor, None]" = None,
) -> Callable:
    """
    Convert a synchronous function to an async function and allow it to be run in a thread pool.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Callable:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            pool, functools.partial(func, *args, **kwargs)
        )
        return result

    return wrapper


def try_rm(files: "Iterable[Union[Path, None]]") -> None:
    """
    Try to remove directories.
    Silently ignores errors for any directory that does not exist.
    """
    for file in files:
        if file is None:
            continue
        try:
            shutil.rmtree(file)
        except NotADirectoryError:
            os.remove(file)
        except FileNotFoundError:
            pass


def write_excel(
    dfs: "Iterable[pd.DataFrame]",
    out_dir: str,
    file_name: str,
    sheet_names: "Iterable[str]",
    write_index: bool = True,
) -> str:
    """
    Write a list of dataframes to an excel file.

    Parameters
    ----------
    dfs : Iterable[pd.DataFrame]
        The dataframes to write.
    out_dir : str
        The directory to write the excel file to.
    file_name : str
        The name of the excel file.
    sheet_names : Iterable[str]
        The names of the sheets to write the dataframes to.
    write_index : bool, optional
        Whether to write the index of the dataframes to the Excel file, by default True

    Returns
    -------
    str
        The path to the written excel file.
    """
    outpath = os.path.join(out_dir, f"{file_name}.xlsx")
    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
        outpath
    ) as writer:
        for df, sheet_name in zip(dfs, sheet_names):  # pylint: disable=invalid-name
            df.to_excel(writer, sheet_name=sheet_name, index=write_index)
    return outpath
