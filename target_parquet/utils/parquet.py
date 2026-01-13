from __future__ import annotations

import datetime
import decimal
import logging

import pyarrow as pa
import pyarrow.parquet as pq

from target_parquet.utils import bytes_to_mb

FIELD_TYPE_TO_PYARROW = {
    "BOOLEAN": pa.bool_(),
    "STRING": pa.string(),
    "ARRAY": pa.string(),
    "INTEGER": pa.int64(),
    "NUMBER": pa.float64(),
    "OBJECT": pa.string(),
}

FORMAT_TO_PYARROW = {
    "date-time": pa.timestamp('us', tz='UTC'),  # Microsecond precision, UTC
    "date": pa.date32(),                         # Date-only values
    "time": pa.time64('us'),                     # Time-only, microsecond precision
}


EXTENSION_MAPPING = {
    "snappy": ".snappy",
    "gzip": ".gz",
    "brotli": ".br",
    "zstd": ".zstd",
    "lz4": ".lz4",
}

logger = logging.getLogger(__name__)


def _field_type_to_pyarrow_field(
    field_name: str, input_types: dict, required_fields: list[str]
) -> pa.Field:
    # Check format field first (for date-time, date, time)
    field_format = input_types.get("format")
    if field_format and field_format in FORMAT_TO_PYARROW:
        nullable = field_name not in required_fields
        # Check if null is in types
        types = input_types.get("type", [])
        types = [types] if isinstance(types, str) else types
        types_uppercase = [item.upper() for item in types]
        if "NULL" in types_uppercase:
            nullable = True
        return pa.field(field_name, FORMAT_TO_PYARROW[field_format], nullable)

    # Existing type-based logic for non-temporal fields
    types = input_types.get("type", [])
    # If type is not defined, check if anyOf is defined
    if not types:
        for any_type in input_types.get("anyOf", []):
            if t := any_type.get("type"):
                if isinstance(t, list):
                    types.extend(t)
                else:
                    types.append(t)
    types = [types] if isinstance(types, str) else types
    types_uppercase = [item.upper() for item in types]
    nullable = "NULL" in types_uppercase or field_name not in required_fields
    if "NULL" in types_uppercase:
        types_uppercase.remove("NULL")
    input_type = next(iter(types_uppercase)) if types_uppercase else ""
    pyarrow_type = FIELD_TYPE_TO_PYARROW.get(input_type, pa.string())
    return pa.field(field_name, pyarrow_type, nullable)


def flatten_schema_to_pyarrow_schema(flatten_schema_dictionary: dict) -> pa.Schema:
    """Function that converts a flatten schema to a pyarrow schema in a defined order.

    E.g:
     dictionary = {
        'properties': {
             'key_1': {'type': ['null', 'integer']},
             'key_2__key_3': {'type': ['null', 'string']},
             'key_2__key_4__key_5': {'type': ['null', 'integer']},
             'key_2__key_4__key_6': {'type': ['null', 'array']}
           }
        }
    By calling the function with the dictionary above as parameter,
    you will get the following structure:
        pa.schema([
             pa.field('key_1', pa.int64()),
             pa.field('key_2__key_3', pa.string()),
             pa.field('key_2__key_4__key_5', pa.int64()),
             pa.field('key_2__key_4__key_6', pa.string())
        ])
    """
    flatten_schema = flatten_schema_dictionary.get("properties", {})
    required_fields = flatten_schema_dictionary.get("required", [])
    return pa.schema(
        [
            _field_type_to_pyarrow_field(
                field_name, field_input_types, required_fields=required_fields
            )
            for field_name, field_input_types in flatten_schema.items()
        ]
    )


def _convert_decimal(value):
    """Convert Decimal"""
    if isinstance(value, decimal.Decimal):
        return float(value)
    return value


def _convert_temporal(value, field_type: pa.DataType):
    """Convert RFC3339 string to appropriate Python temporal object.

    Args:
        value: String value in RFC3339 format (or None)
        field_type: Target PyArrow data type

    Returns:
        Converted temporal object or None
    """
    if value is None or value == "":
        return None

    if not isinstance(value, str):
        return value  # Already converted or non-string type

    try:
        if pa.types.is_timestamp(field_type):
            # Parse RFC3339 datetime string
            # Replace 'Z' suffix with '+00:00' for Python compatibility
            dt = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
            # Ensure timezone-aware, assume UTC if naive
            if dt.tzinfo is None:
                return dt.replace(tzinfo=datetime.timezone.utc)
            else:
                # Convert to UTC
                return dt.astimezone(datetime.timezone.utc)

        elif pa.types.is_date(field_type):
            # Parse date-only value
            return datetime.datetime.fromisoformat(value).date()

        elif pa.types.is_time(field_type):
            # Parse time-only value (prepend dummy date for parsing)
            return datetime.datetime.fromisoformat(f'1970-01-01T{value}').time()

    except (ValueError, AttributeError) as e:
        logger.error(f"Failed to parse temporal value '{value}': {e}")
        raise ValueError(f"Invalid temporal format for field with type {field_type}: {value}") from e

    return value


def create_pyarrow_table(list_dict: list[dict], schema: pa.Schema) -> pa.Table:
    """Create a pyarrow Table from a python list of dict."""
    data = {}
    for field in schema:
        field_name = field.name
        field_type = field.type
        # Apply conversions for each field based on its type
        column_data = []
        for row in list_dict:
            value = row.get(field_name)
            # Convert decimals
            value = _convert_decimal(value)
            # Convert temporal types
            if pa.types.is_temporal(field_type):
                value = _convert_temporal(value, field_type)
            column_data.append(value)
        data[field_name] = column_data

    return pa.table(data).cast(schema)


def concat_tables(
    records: list[dict], pyarrow_table: pa.Table, pyarrow_schema: pa.Schema
) -> pa.Table:
    """Create a dataframe from records and concatenate with the existing one."""
    if not records:
        return pyarrow_table
    new_table = create_pyarrow_table(records, pyarrow_schema)
    return pa.concat_tables([pyarrow_table, new_table]) if pyarrow_table else new_table


def write_parquet_file(
    table: pa.Table,
    path: str,
    compression_method: str = "gzip",
    basename_template: str | None = None,
    partition_cols: list[str] | None = None,
) -> None:
    """Write a pyarrow table to a parquet file."""
    pq.write_to_dataset(
        table,
        root_path=path,
        compression=compression_method,
        partition_cols=partition_cols or None,
        use_threads=True,
        use_legacy_dataset=False,
        basename_template=f"{basename_template}{EXTENSION_MAPPING[compression_method.lower()]}.parquet"
        if basename_template
        else None,
    )


def get_pyarrow_table_size(table: pa.Table) -> float:
    """Return the size of a pyarrow table in MB."""
    return bytes_to_mb(table.nbytes)
