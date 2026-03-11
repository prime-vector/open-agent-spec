# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""JSON schema type → Python type hint mapping for generated task signatures.

Generated agent.py always includes ``from typing import Optional, Any, Dict, List``
(see data_preparation._prepare_imports), so Dict[str, Any] / List[Any] in signatures
are valid. If that import block ever changes, keep this mapping in sync or switch
to PEP 585 forms (dict[str, Any]) and ensure the template imports Any only.
"""



def map_type_to_python(schema_type: str) -> str:
    """Map JSON schema type string to Python type hint string (used in generated code)."""
    return {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        # Rely on generated imports: Dict/List/Any (see agent template preamble).
        "object": "Dict[str, Any]",
        "array": "List[Any]",
    }.get(schema_type, "Any")
