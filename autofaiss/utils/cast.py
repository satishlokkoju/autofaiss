" function to cast variables in others "

import numbers
import re
from math import floor
from typing import Union, Any, Dict, List

import faiss
import numpy as np


def cast_memory_to_bytes(memory_string: str) -> float:
    """
    Parse a memory string and returns the number of bytes
    >>> cast_memory_to_bytes("16B")
    16
    >>> cast_memory_to_bytes("16G") == 16*1024*1024*1024
    True
    """

    conversion = {unit: (2**10) ** i for i, unit in enumerate("BKMGTPEZ")}

    number_match = r"([0-9]*\.[0-9]+|[0-9]+)"
    unit_match = "("
    for unit in conversion:
        if unit != "B":
            unit_match += unit + "B|"
    for unit in conversion:
        unit_match += unit + "|"
    unit_match = unit_match[:-1] + ")"

    matching_groups = re.findall(number_match + unit_match, memory_string, re.IGNORECASE)

    if matching_groups and len(matching_groups) == 1 and "".join(matching_groups[0]) == memory_string:
        group = matching_groups[0]
        return float(group[0]) * conversion[group[1][0].upper()]

    raise ValueError(f"Unknown format for memory string: {memory_string}")


def cast_bytes_to_memory_string(num_bytes: float) -> str:
    """
    Cast a number of bytes to a readable string
    """

    suffix = "B"
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_bytes) < 1024.0:
            return "%3.1f%s%s" % (num_bytes, unit, suffix)  # pylint: disable=consider-using-f-string
        num_bytes /= 1024.0
    return "%.1f%s%s" % (num_bytes, "Y", suffix)  # pylint: disable=consider-using-f-string


def to_faiss_metric_type(metric_type: Union[str, int]) -> int:
    """convert metric_type string/enum to faiss enum of the distance metric"""

    if metric_type in ["ip", "IP", faiss.METRIC_INNER_PRODUCT]:
        return faiss.METRIC_INNER_PRODUCT
    elif metric_type in ["l2", "L2", faiss.METRIC_L2]:
        return faiss.METRIC_L2
    else:
        raise ValueError("Metric currently not supported")


def to_readable_time(seconds: float, rounding: bool = False) -> str:
    """cast time in seconds to readable string"""

    initial_seconds = seconds

    hours = int(floor(seconds // 3600))
    seconds -= 3600 * hours
    minutes = int(floor(seconds // 60))
    seconds -= 60 * minutes

    if rounding:
        if hours:
            return f"{initial_seconds/3600:3.1f} hours"
        if minutes:
            return f"{initial_seconds/60:3.1f} minutes"
        return f"{initial_seconds:3.1f} seconds"

    time_str = ""
    if hours:
        time_str += f"{hours:d} hours "
    if hours or minutes:
        time_str += f"{minutes:d} minutes "
    time_str += f"{seconds:.2f} seconds"

    return time_str


def convert_numpy_types_to_python(obj: Any) -> Any:
    """
    Recursively convert numpy numeric types to native Python types for JSON serialization.

    Parameters:
    -----------
    obj: Any
        Object to convert. Can be dict, list, numpy scalar, or other types.

    Returns:
    --------
    Any: Object with numpy types converted to Python types.
    """
    if isinstance(obj, np.ndarray):
        # For arrays, convert to list (assuming 0-d or rarely used in JSON)
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types_to_python(item) for item in obj)
    else:
        return obj
