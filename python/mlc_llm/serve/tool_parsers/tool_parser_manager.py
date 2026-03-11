from typing import Dict, List

class ToolParserManager:
    """Tool parser manager for registering and retrieving tool parsers."""

    _parsers: Dict[str, type] = {}

    @classmethod
    def register_module(cls, name: str):
        """Register a tool parser class."""
        def decorator(tool_parser_class: type):
            cls._parsers[name] = tool_parser_class
            return tool_parser_class
        return decorator

    @classmethod
    def get_parser(cls, name: str) -> type:
        """Get a tool parser class by name."""
        return cls._parsers.get(name)

    @classmethod
    def get_parser_names(cls) -> List[str]:
        """Get all registered parser names."""
        return list(cls._parsers.keys())
