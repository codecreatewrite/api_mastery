#!/usr/bin/env python3
"""
Lesson: api_key_client
Created: 2025-09-09T12:38:57+01:00
Project: api_mastery
Template: auth (client credentials style)
"""

import requests
import os
from typing import Optional, Dict

class APIKeyClient:
    """
    Handle API key authentication patterns
    """
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def query_parameter_auth(self, endpoint: str, params: Dict = None) -> Dict:
        """API key in URL parameters (less secure)"""
        if params is None:
            params = {}
        
        params['api_key'] = self.api_key  # or 'key', 'appid', etc.
        
        response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
        return self._handle_response(response)
    
    def header_auth(self, endpoint: str, params: Dict = None) -> Dict:
        """API key in headers (more secure)"""
        headers = {
            'X-API-Key': self.api_key,  # Common patterns
            # 'Authorization': f'Bearer {self.api_key}',
            # 'X-RapidAPI-Key': self.api_key,
        }
        
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            headers=headers
        )
        return self._handle_response(response)
    
    def custom_header_auth(self, endpoint: str, header_name: str, params: Dict = None) -> Dict:
        """Flexible header-based auth for different APIs"""
        headers = {header_name: self.api_key}
        
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            headers=headers
        )
        return self._handle_response(response)
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """Standard response handling"""
        if response.status_code == 200:
            return {'success': True, 'data': response.json()}
        elif response.status_code == 401:
            return {'success': False, 'error': 'Invalid API key'}
        elif response.status_code == 403:
            return {'success': False, 'error': 'API key lacks permissions'}
        elif response.status_code == 429:
            return {'success': False, 'error': 'Rate limit exceeded'}
        else:
            return {'success': False, 'error': f'HTTP {response.status_code}'}

# Example: OpenWeatherMap API
weather_client = APIKeyClient("your-openweather-key", "http://api.openweathermap.org/data/2.5")
result = weather_client.query_parameter_auth("weather", {"q": "London", "units": "metric"})
