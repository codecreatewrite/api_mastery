#!/usr/bin/env python3
"""
Lesson: city_weather_comparison_tool
Created: 2025-08-29T02:17:02+01:00
Project: api_mastery
Template: script
"""
import os
import sys
import json
import time
from typing import Any, Dict

try:
    import requests  # pip install requests
except Exception as e:
    print("[setup] Missing dependency: requests — pip install requests", file=sys.stderr)
    raise

BASE_URL = os.getenv("BASE_URL", "https://httpbin.org")
API_KEY = os.getenv("API_KEY", "")
DEFAULT_HEADERS = {"User-Agent": "api-mastery/lesson", **({"Authorization": f"Bearer {API_KEY}"} if API_KEY else {})}

def get_json(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=DEFAULT_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    print("GET /get →", json.dumps(get_json("get", {"hello": "world"}), indent=2))
