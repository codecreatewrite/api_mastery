#!/usr/bin/env python3
"""
Lesson: bearer_token_client
Created: 2025-09-09T19:37:17+01:00
Project: api_mastery
Template: script
"""
import requests
import json
from datetime import datetime, timedelta
from typing import Optional

class BearerTokenClient:
    """
    Handle Bearer token authentication (JWT, OAuth tokens, etc.)
    """
    
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.token_expires = None
        self.session = requests.Session()
        
        if token:
            self._set_auth_header(token)
    
    def _set_auth_header(self, token: str):
        """Set Authorization header for all requests"""
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def login_with_credentials(self, username: str, password: str, login_endpoint: str = "auth/login") -> Dict:
        """Login to get bearer token"""
        login_data = {
            'username': username,
            'password': password
        }
        
        response = self.session.post(
            f"{self.base_url}/{login_endpoint}",
            json=login_data
        )
        
        if response.status_code == 200:
            token_data = response.json()
            self.token = token_data.get('access_token') or token_data.get('token')
            
            # Handle token expiration if provided
            if 'expires_in' in token_data:
                self.token_expires = datetime.now() + timedelta(seconds=token_data['expires_in'])
            
            self._set_auth_header(self.token)
            return {'success': True, 'token': self.token}
        else:
            return {'success': False, 'error': 'Login failed'}
    
    def is_token_expired(self) -> bool:
        """Check if token is expired"""
        if not self.token_expires:
            return False
        return datetime.now() >= self.token_expires
    
    def refresh_token_if_needed(self, refresh_endpoint: str = "auth/refresh"):
        """Refresh token if it's expired"""
        if self.is_token_expired():
            # Implementation depends on API's refresh mechanism
            print("⚠️  Token expired, refreshing...")
            # This would typically involve calling a refresh endpoint
    
    def authenticated_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated request with automatic token refresh"""
        self.refresh_token_if_needed()
        
        response = self.session.request(
            method,
            f"{self.base_url}/{endpoint}",
            **kwargs
        )
        
        if response.status_code == 401:
            return {'success': False, 'error': 'Token invalid or expired'}
        elif response.status_code == 403:
            return {'success': False, 'error': 'Insufficient permissions'}
        elif response.status_code == 200:
            return {'success': True, 'data': response.json()}
        else:
            return {'success': False, 'error': f'HTTP {response.status_code}'}

if __name__ == '__main__':
    # Example usage
    client = BearerTokenClient("https://api.example.com")
    login_result = client.login_with_credentials("username", "password")
    if login_result['success']:
        data = client.authenticated_request("GET", "user/profile")
