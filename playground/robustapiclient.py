#!/usr/bin/env python3
"""
Lesson: robustapiclient
Created: 2025-09-01T22:04:33+01:00
Project: api_mastery
Template: script
"""

import requests
import time
from functools import wraps

# ----------------------------
# Decorator: retry_on_failure
# ----------------------------
def retry_on_failure(method):
    """
    Retries the decorated instance method on network-related exceptions.
    It expects `self` to provide `.max_retries`.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        last_exception = None
        # total attempts = 1 initial + self.max_retries retries
        for attempt in range(self.max_retries + 1):
            try:
                return method(self, *args, **kwargs)
            except requests.exceptions.Timeout as e:
                last_exception = e
                print(f"Timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                print(f"Connection error on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                last_exception = e
                print(f"Request error on attempt {attempt + 1}: {e}")

            # exponential backoff (2, 4, 8, ...)
            if attempt < self.max_retries:
                wait_time = 2 ** (attempt + 1)
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        # no attempts left -> raise the last network exception
        raise last_exception

    return wrapper


class RobustAPIClient:
    """
    A robust API client:
    - Reuses a Session
    - Provides timeouts
    - Retries on network exceptions
    - Normalizes responses into dicts like {'success': bool, 'data': ..., 'error': ...}
    """

    def __init__(self, base_url, timeout=30, max_retries=3):
        # Normalize base_url to avoid trailing slash problems
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    @retry_on_failure
    def _make_request(self, method, endpoint, **kwargs):
        """
        Internal: build URL, set default timeout, perform HTTP request, and process response.
        - `method` is a string: 'GET', 'POST', 'PUT', 'DELETE'
        - `endpoint` is the path part (with or without leading '/')
        - `**kwargs` are forwarded to requests (params, json, data, headers, timeout, etc.)
        """
        # 1. Build URL safely
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        # 2. Ensure there's always a timeout unless the caller provided one
        kwargs.setdefault('timeout', self.timeout)

        # Example: after setdefault, kwargs might look like:
        # {'params': {'q': 'x'}, 'timeout': 30}

        # 3. Make the actual HTTP call using Session (returns a Response)
        response = self.session.request(method, url, **kwargs)

        # 4. Convert raw Response into normalized dict
        return self._process_response(response)

    def _safe_json(self, response):
        """Try to parse JSON; return None on failure instead of raising."""
        try:
            return response.json()
        except ValueError:
            return None

    def _process_response(self, response):
        """
        Convert Response -> normalized dict.
        This method *does not* raise; it returns a dictionary describing success/failure.
        """
        status = response.status_code

        if status == 200:
            return {'success': True, 'data': self._safe_json(response)}

        elif status == 201:
            return {'success': True, 'data': self._safe_json(response), 'created': True}

        elif status == 204:
            return {'success': True, 'data': None}

        elif status == 400:
            return {'success': False, 'error': 'Bad Request', 'details': response.text}

        elif status == 401:
            return {'success': False, 'error': 'Unauthorized - Check your API key'}

        elif status == 403:
            return {'success': False, 'error': 'Forbidden - Insufficient permissions'}

        elif status == 404:
            return {'success': False, 'error': 'Resource not found'}

        elif status == 429:
            retry_after = response.headers.get('Retry-After', 60)
            try:
                retry_after = int(float(retry_after))
            except (TypeError, ValueError):
                retry_after = 60
            return {'success': False, 'error': 'Rate limited', 'retry_after': retry_after}

        elif 500 <= status < 600:
            return {'success': False, 'error': 'Server error', 'status_code': status}

        else:
            return {'success': False, 'error': f'Unexpected status code: {status}', 'details': response.text}

    # --- convenience methods accept **kwargs so callers can pass headers/timeout/etc. ---
    def get(self, endpoint, params=None, **kwargs):
        return self._make_request('GET', endpoint, params=params, **kwargs)

    def post(self, endpoint, data=None, json=None, **kwargs):
        return self._make_request('POST', endpoint, data=data, json=json, **kwargs)

    def put(self, endpoint, data=None, json=None, **kwargs):
        return self._make_request('PUT', endpoint, data=data, json=json, **kwargs)

    def delete(self, endpoint, **kwargs):
        return self._make_request('DELETE', endpoint, **kwargs)


client = RobustAPIClient("https://jsonplaceholder.typicode.com/")

# 1) GET an existing resource (server returns 200 with JSON body)
result = client.get("posts/1")
print(result)

# 2) GET a non-existent resource (server returns 404)
result = client.get("posts/999999")
print(result)

# 3) POST that created resource (server returns 201 + JSON)
payload = {'title': 'hello', 'body': 'x', 'userId': 1}
result = client.post("posts", json=payload)
print(result)

# 4) DELETE (server may return 200 or 204 depending on API)
result = client.delete("posts/1")
print(result)

# 5) Server says "Too Many Requests" with Retry-After header
# Suppose server returns status 429 and header Retry-After: "120"
result = client.get("rate_limited_endpoint")
print(result)

# 6) Server returns 500
result = client.get("internal_error")
print(result)

