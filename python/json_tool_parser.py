# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the MLC LLM project
"""JSON tool parser for MLC LLM."""

import json
from typing import Any, Callable, Dict, Optional


class JsonToolParser:
    """A parser for JSON tool definitions."""

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        transform_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """Initialize the JSON tool parser.

        Args:
            schema: Optional schema for validating JSON objects.
            transform_fn: Optional transformation function to apply to parsed JSON.
        """
        self.schema = schema
        self.transform_fn = transform_fn

    @staticmethod
    def parse_to_json_object(json_str: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON string to a Python object.

        Args:
            json_str: JSON string to parse.

        Returns:
            Parsed Python object (dict) or None if parsing fails.
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def render_to_json_string(obj: Any) -> str:
        """Render a Python object to a JSON string.

        Args:
            obj: Python object to render.

        Returns:
            JSON string representation.
        """
        return json.dumps(obj, ensure_ascii=False)

    def parse_and_validate(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Parse and validate a JSON string against the schema.

        Args:
            json_str: JSON string to parse and validate.

        Returns:
            Parsed Python object if valid, None otherwise.
        """
        parsed_obj = self.parse_to_json_object(json_str)
        if parsed_obj is None:
            return None

        # Validate against schema if provided
        if self.schema is not None:
            if not self._validate_against_schema(parsed_obj, self.schema):
                return None

        return parsed_obj

    def parse_transform_and_render(self, json_str: str) -> Optional[str]:
        """Parse, transform, and render a JSON string.

        Args:
            json_str: JSON string to process.

        Returns:
            Rendered JSON string if successful, None otherwise.
        """
        parsed_obj = self.parse_to_json_object(json_str)
        if parsed_obj is None:
            return None

        # Apply transformation if provided
        if self.transform_fn is not None:
            parsed_obj = self.transform_fn(parsed_obj)

        return self.render_to_json_string(parsed_obj)

    def _validate_against_schema(
        self,
        obj: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> bool:
        """Validate an object against a schema.

        Args:
            obj: Object to validate.
            schema: Schema to validate against.

        Returns:
            True if valid, False otherwise.
        """
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in obj:
                return False

        # Validate each field
        for field_name, field_schema in schema.items():
            if field_name not in obj:
                continue

            field_type = field_schema.get("type")
            if field_type:
                if not self._check_type(obj[field_name], field_type):
                    return False

        return True

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches an expected type.

        Args:
            value: Value to check.
            expected_type: Expected type as string.

        Returns:
            True if type matches, False otherwise.
        """
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "None": type(None),
        }

        expected = type_mapping.get(expected_type)
        if expected is None:
            return True  # Unknown type, skip validation

        # Special case: bool is a subclass of int in Python
        if expected_type == "int" and isinstance(value, bool):
            return False

        return isinstance(value, expected)
