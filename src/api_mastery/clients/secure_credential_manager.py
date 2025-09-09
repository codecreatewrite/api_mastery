#!/usr/bin/env python3
"""
Lesson: secure_credential_manager
Created: 2025-09-09T19:51:46+01:00
Project: api_mastery
Template: script
"""
import os
import json
import keyring  # pip install keyring
from pathlib import Path
from typing import Dict, Optional
import configparser

class SecureCredentialManager:
    """
    Professional credential management system
    """
    
    def __init__(self, app_name: str = "api_client"):
        self.app_name = app_name
        self.config_file = Path.home() / f".{app_name}_config.ini"
        self.config = configparser.ConfigParser()
        
        # Load existing configuration
        if self.config_file.exists():
            self.config.read(self.config_file)
    
    def store_api_key(self, service_name: str, api_key: str, use_keyring: bool = True):
        """Store API key securely"""
        if use_keyring:
            # Store in system keyring (most secure)
            keyring.set_password(self.app_name, service_name, api_key)
            
            # Store metadata in config file (not the actual key)
            if 'credentials' not in self.config:
                self.config.add_section('credentials')
            self.config['credentials'][service_name] = 'keyring'
        else:
            # Store in config file (less secure, for development only)
            if 'api_keys' not in self.config:
                self.config.add_section('api_keys')
            self.config['api_keys'][service_name] = api_key
        
        self._save_config()
    
    def get_api_key(self, service_name: str) -> Optional[str]:
        """Retrieve API key from secure storage"""
        # Try environment variable first
        env_key = f"{service_name.upper()}_API_KEY"
        if env_key in os.environ:
            return os.environ[env_key]
        
        # Try keyring
        if ('credentials' in self.config and 
            service_name in self.config['credentials'] and
            self.config['credentials'][service_name] == 'keyring'):
            return keyring.get_password(self.app_name, service_name)
        
        # Try config file (development only)
        if ('api_keys' in self.config and 
            service_name in self.config['api_keys']):
            return self.config['api_keys'][service_name]
        
        return None
    
    def setup_credentials_interactive(self):
        """Interactive setup for API credentials"""
        print("🔐 API Credential Setup")
        print("=" * 30)
        
        services = [
            'openweather',
            'github',
            'newsapi',
            'weatherapi'
        ]
        
        for service in services:
            existing = self.get_api_key(service)
            if existing:
                update = input(f"{service} API key exists. Update? (y/N): ").lower()
                if update != 'y':
                    continue
            
            api_key = input(f"Enter {service} API key (or press Enter to skip): ").strip()
            if api_key:
                self.store_api_key(service, api_key)
                print(f"✅ {service} API key stored securely")
    
    def list_stored_credentials(self):
        """List all stored credentials (without revealing keys)"""
        print("🔑 Stored Credentials:")
        
        # Check environment variables
        env_keys = [key for key in os.environ.keys() if key.endswith('_API_KEY')]
        for key in env_keys:
            service = key.replace('_API_KEY', '').lower()
            print(f"  ✅ {service} (environment variable)")
        
        # Check keyring
        if 'credentials' in self.config:
            for service in self.config['credentials']:
                if self.config['credentials'][service] == 'keyring':
                    print(f"  ✅ {service} (system keyring)")
        
        # Check config file
        if 'api_keys' in self.config:
            for service in self.config['api_keys']:
                print(f"  ⚠️  {service} (config file - less secure)")
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        
        # Set restrictive permissions (Unix-like systems)
        try:
            self.config_file.chmod(0o600)  # Read/write for owner only
        except:
            pass

if __name__ == '__main__':
    # Usage example
    credential_manager = SecureCredentialManager("weather_dashboard")

    # Interactive setup (run once)
    # credential_manager.setup_credentials_interactive()

    # In your application code
    openweather_key = credential_manager.get_api_key('openweather')
    github_token = credential_manager.get_api_key('github')
