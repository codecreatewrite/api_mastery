#!/usr/bin/env python3
"""
Lesson: requests_wraps_time
Created: 2025-08-31T19:35:10+01:00
Project: api_mastery
Template: script
"""
"""
This script demonstrates the most common patterns you’ll use 
when working with Python decorators, `time`, `functools.wraps`, 
and `requests`.

The goal: Give you a small "toolbox" you’ll actually reuse in 
real projects — fetching data from APIs, timing code, retrying 
failed requests, caching results, etc.
"""

# === 1. Imports ===
import time  # for timing execution and sleep/delays
import requests  # for making HTTP requests (API calls, scraping, etc.)
from functools import wraps  # for writing proper decorators


# === 2. Pattern #1: Simple decorator ===
def timed(func):
    """
    Decorator that measures how long a function takes to run.

    ✅ Most common pattern:
    - Wraps a function
    - Adds behavior (timing/logging) before/after
    - Returns result unchanged
    """

    @wraps(func)  # makes sure metadata (name, docstring) stays intact
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # high-resolution start time
        try:
            return func(*args, **kwargs)  # run the original function
        finally:
            end = time.perf_counter()
            print(f"{func.__name__} took {end - start:.3f} seconds")

    return wrapper


# === 3. Pattern #2: Decorator with arguments (decorator factory) ===
def retry(max_attempts=3, delay=1):
    """
    Retry decorator for unreliable operations (like network requests).

    Usage:
        @retry(max_attempts=5, delay=2)
        def fetch_data(): ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)  # wait before retrying
            raise last_exc  # re-raise the last error after retries fail

        return wrapper

    return decorator


# === 4. Pattern #3: Stateful decorator (with attributes) ===
def count_calls(func):
    """
    Counts how many times a function was called.

    ✅ Useful for debugging, limiting, analytics.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"{func.__name__} has been called {wrapper.calls} times")
        return func(*args, **kwargs)

    wrapper.calls = 0  # initialize state
    return wrapper


# === 5. Combine everything: Fetch data from an API ===

@timed  # measure how long it takes
@retry(max_attempts=3, delay=2)  # retry if request fails
@count_calls  # track how many times function is called
def fetch_json(url):
    """
    Fetch JSON data from a URL using requests.
    Demonstrates using decorators in a realistic workflow.
    """
    response = requests.get(url, timeout=5)  # make request
    response.raise_for_status()  # raise error if bad status (4xx, 5xx)
    return response.json()


# === 6. Example usage ===
if __name__ == "__main__":
    # Public API that always works: returns fake posts
    url = "https://jsonplaceholder.typicode.com/posts/1"

    # Call the decorated function
    data = fetch_json(url)

    print("\nData received from API:")
    print(data)

    # Call again to see call counter increase
    fetch_json(url)
