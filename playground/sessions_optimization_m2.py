#!/usr/bin/env python3
"""
Lesson: sessions_optimization_m2
Created: 2025-09-03T23:27:47+01:00
Project: api_mastery
Template: script
"""
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ❌ INEFFICIENT: Creates new connection each time
def slow_multiple_requests():
    start_time = time.time()

    for i in range(5):
        response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
        print(f"Request {i+1}: {response.status_code}")

    end_time = time.time()
    print(f"Time without session: {end_time - start_time:.2f} seconds")

# ✅ EFFICIENT: Reuses connection
def fast_multiple_requests():
    start_time = time.time()

    with requests.Session() as session:
        for i in range(5):
            response = session.get("https://jsonplaceholder.typicode.com/posts/1")
            print(f"Request {i+1}: {response.status_code}")

    end_time = time.time()
    print(f"Time with session: {end_time - start_time:.2f} seconds")

# Test both approaches
print("Testing connection efficiency:")
slow_multiple_requests()
print()
fast_multiple_requests()

#--------------------------------------------
#Advanced version
#--------------------------------------------
class AdvancedAPIClient:
    """
    Professional-grade API client with advanced session configuration
    """

    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.session = self._create_session(api_key)

    def _create_session(self, api_key):
        """Create optimized session with retry strategy"""
        session = requests.Session()

        # Set default headers
        session.headers.update({
            'User-Agent': 'PythonAPIClient/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        # Add API key if provided
        if api_key:
            session.headers.update({'Authorization': f'Bearer {api_key}'})

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,                    # Total number of retries
            backoff_factor=1,           # Wait time between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry
            allowed_methods=["HEAD", "GET", "OPTIONS"]    # Methods to retry
        )

        # Create adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)

        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def close(self):
        """Clean up session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Usage with context manager (automatically closes session)
with AdvancedAPIClient("https://api.example.com", "your-api-key") as client:
    # Your API calls here
    pass
