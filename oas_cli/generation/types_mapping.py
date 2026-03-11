# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""JSON schema type → Python type hint mapping for generated task signatures."""


def map_type_to_python(t):
    """Map JSON schema type string to Python type hint (used by task codegen)."""
    return {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "object": "Dict[str, Any]",
        "array": "List[Any]",
    }.get(t, "Any")
