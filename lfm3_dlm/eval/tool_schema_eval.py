"""XML `<tool_call>` and JSON-schema validators for structured-extraction eval."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

# LFM3 tool-call grammar: <tool_call><name>..</name><arguments>{json}</arguments></tool_call>
_REQUIRED_TAG_KEYS = ("name", "arguments")


def validate_tool_call_xml(text: str) -> dict:
    """Validate an LFM3 XML `<tool_call>` payload.

    Returns `{"ok": bool, "errors": [...], "name": str|None,
    "arguments": dict|None}`.
    """
    errors: list[str] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        return {"ok": False, "errors": [f"malformed XML: {e}"],
                "name": None, "arguments": None}
    if root.tag != "tool_call":
        errors.append(f"root tag must be 'tool_call', got {root.tag!r}")
    for key in _REQUIRED_TAG_KEYS:
        if root.find(key) is None or not (root.findtext(key) or "").strip():
            errors.append(f"missing or empty <{key}> element")
    name = (root.findtext("name") or "").strip() or None
    args = None
    arg_text = (root.findtext("arguments") or "").strip()
    if arg_text:
        try:
            args = json.loads(arg_text)
            if not isinstance(args, dict):
                errors.append("<arguments> must decode to a JSON object")
                args = None
        except json.JSONDecodeError as e:
            errors.append(f"<arguments> is not valid JSON: {e}")
    return {"ok": not errors, "errors": errors,
            "name": name, "arguments": args}


_TYPE_MAP = {"object": dict, "array": list, "string": str,
             "number": (int, float), "integer": int, "boolean": bool}


def validate_tool_call_json(payload, schema: dict) -> dict:
    """Minimal JSON-schema check (type + required + properties + enum).

    `payload` may be a dict or a JSON string. Covers what BFCL-XML /
    structured-extraction needs; not a full Draft-7 implementation.
    """
    errors: list[str] = []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            return {"ok": False, "errors": [f"not valid JSON: {e}"]}

    def check(value, sch, path):
        t = sch.get("type")
        if t:
            py_t = _TYPE_MAP.get(t)
            if py_t is not None and not isinstance(value, py_t):
                errors.append(f"{path or '<root>'}: expected type {t}")
                return
            # bool is a subclass of int — reject it for number/integer
            if t in ("integer", "number") and isinstance(value, bool):
                errors.append(f"{path or '<root>'}: expected type {t}")
                return
        if "enum" in sch and value not in sch["enum"]:
            errors.append(f"{path or '<root>'}: {value!r} not in enum")
        if t == "object" or isinstance(value, dict):
            for req in sch.get("required", []):
                if not isinstance(value, dict) or req not in value:
                    errors.append(f"{path or '<root>'}: missing key {req!r}")
            props = sch.get("properties", {})
            if isinstance(value, dict):
                if sch.get("additionalProperties") is False:
                    for k in value:
                        if k not in props:
                            errors.append(
                                f"{path or '<root>'}: unexpected key {k!r}")
                for k, sub in props.items():
                    if k in value:
                        check(value[k], sub,
                              f"{path}.{k}" if path else k)

    check(payload, schema, "")
    return {"ok": not errors, "errors": errors}
