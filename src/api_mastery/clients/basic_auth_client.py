#!/usr/bin/env python3
"""
Lesson: basic_auth_client
Created: 2025-09-09T19:27:00+01:00
Project: api_mastery
Template: auth (client credentials style)
"""
import requests
from requests.auth import HTTPBasicAuth
import base64

class BasicAuthClient:
    """
    Handle HTTP Basic Authentication
    """
    
    def __init__(self, username: str, password: str, base_url: str):
        self.username = username
        self.password = password
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def requests_built_in_auth(self, endpoint: str) -> Dict:
        """Use requests built-in Basic Auth"""
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            auth=HTTPBasicAuth(self.username, self.password)
        )
        return self._handle_response(response)
    
    def manual_basic_auth(self, endpoint: str) -> Dict:
        """Manual Basic Auth header creation"""
        # Create base64-encoded credentials
        credentials = f"{self.username}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {encoded_credentials}'
        }
        
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            headers=headers
        )
        return self._handle_response(response)
    
    def _handle_response(self, response: requests.Response) -> Dict:
        if response.status_code == 200:
            return {'success': True, 'data': response.json()}
        elif response.status_code == 401:
            return {'success': False, 'error': 'Invalid credentials'}
        else:
            return {'success': False, 'error': f'HTTP {response.status_code}'}

if __name__ == '__main__':
    # Example usage
    client = BasicAuthClient("username", "password", "https://api.example.com")
    result = client.requests_built_in_auth("protected-endpoint")
