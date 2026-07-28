"""
Utility functions for agent tools.
"""

import json
import re


def format_tool_descriptions(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        name = tool["name"]
        description = tool["description"]
        properties = tool["input_schema"].get("properties", {})
        required = tool["input_schema"].get("required", [])

        params = []
        for param, schema in properties.items():
            req = "" if param in required else " (optional)"
            params.append(f"  - {param}{req}: {schema.get('description', '')}")

        param_str = "\n".join(params) if params else "  - no parameters"
        lines.append(f"- {name}: {description}\n{param_str}")

    return "\n\n".join(lines)


format_tools_for_prompt = format_tool_descriptions


def parse_json(text: str, reason: str | None = None) -> dict | None:
    # Return None if the response is empty.
    if not text:
        return None

    # Remove leading/trailing whitespace from the response.
    cleaned = text.strip()

    # Remove Markdown code block markers if the JSON is wrapped in them.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    # Try parsing the cleaned text directly as JSON.
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            # Only validate key presence if reason is explicitly provided.
            if reason is None or reason in data:
                return data
    except Exception:
        pass

    # As a fallback, extract the first JSON object from the response.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                if reason is None or reason in data:
                    return data
        except Exception:
            pass

    return None
