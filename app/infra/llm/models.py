from dataclasses import dataclass
from typing import Literal

@dataclass
class HopModel:
    queries: list
    reason: Literal['multi-hop', 'single-hop']

hop_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "query_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "reason": {
                            "type": "string",
                            "enum": ["multi-hop", "single-hop"]
                        }
                    },
                    "required": ["queries", "reason"],
                    "additionalProperties": False
                }
            }
        }