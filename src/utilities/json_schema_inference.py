"""Generate a JSON schema from a sample of JSON objects. Note that the result does not adhere to any specification."""

import json

def update_schema_with_object(schema: dict, json_obj: dict) -> dict:
    """Update the schema with data from the given JSON object."""
    for key, value in json_obj.items():
        if key not in schema:
            if isinstance(value, dict):
                schema[key] = {
                    "type": "object",
                    "properties": update_schema_with_object({}, value),
                }
            elif (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], dict)
            ):
                schema[key] = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": update_schema_with_object({}, value[0]),
                    },
                }
            else:
                schema[key] = {"type": type(value).__name__, "example": value}
        elif schema[key]["type"] == "object" and isinstance(value, dict):
            schema[key]["properties"] = update_schema_with_object(
                schema[key]["properties"], value
            )
        elif (
            schema[key]["type"] == "array"
            and isinstance(value, list)
            and len(value) > 0
            and isinstance(value[0], dict)
        ):
            schema[key]["items"]["properties"] = update_schema_with_object(
                schema[key]["items"]["properties"], value[0]
            )

    return schema

def generate_json_schema(input_file: str) -> dict:
    """Generate a JSON schema from a sample of JSON objects."""
    schema = {}
    with open(input_file, "r", encoding="utf-8") as file:
        jsons = json.load(file)
        for json_obj in jsons:
            update_schema_with_object(schema, json_obj)
    return schema

def main():
    """Generate a JSON schema from a sample of JSON objects."""
    input_file = "data.json"
    schema = generate_json_schema(input_file)
    with open("schema.json", "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=4)
