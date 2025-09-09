#!/usr/bin/env python3
"""
Lesson: environconfig
Created: 2025-09-09T19:54:45+01:00
Project: api_mastery
Template: script
"""
import os
from dotenv import load_dotenv  # pip install python-dotenv

# Load environment variables from .env file
load_dotenv()

class EnvironmentConfig:
    """Configuration from environment variables"""
    
    def __init__(self):
        self.openweather_key = self._get_required_env('OPENWEATHER_API_KEY')
        self.github_token = self._get_optional_env('GITHUB_TOKEN')
        self.news_api_key = self._get_optional_env('NEWS_API_KEY')
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _get_optional_env(self, key: str) -> Optional[str]:
        """Get optional environment variable"""
        return os.getenv(key)
    
    def validate_all(self) -> Dict[str, bool]:
        """Check which API keys are available"""
        return {
            'openweather': bool(self.openweather_key),
            'github': bool(self.github_token),
            'news': bool(self.news_api_key),
        }

if __name__ == '__main__':
    # Usage
    config = EnvironmentConfig()
    print("Available APIs:", config.validate_all())
