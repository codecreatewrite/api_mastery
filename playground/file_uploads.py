#!/usr/bin/env python3
"""
Lesson: file_uploads
Created: 2025-09-03T23:41:14+01:00
Project: api_mastery
Template: script
"""
import requests
import json

class MultiFormatAPIClient:
    """
    Handle different data formats and file uploads
    """
    
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
    
    def send_json_data(self, endpoint, data):
        """Send JSON data"""
        response = self.session.post(
            f"{self.base_url}/{endpoint}",
            json=data,  # Automatically sets Content-Type: application/json
            headers={'Accept': 'application/json'}
        )
        return response.json()
    
    def send_form_data(self, endpoint, data):
        """Send form-encoded data"""
        response = self.session.post(
            f"{self.base_url}/{endpoint}",
            data=data,  # Content-Type: application/x-www-form-urlencoded
            headers={'Accept': 'application/json'}
        )
        return response.json()
    
    def upload_file(self, endpoint, file_path, additional_data=None):
        """Upload a file with optional additional data"""
        files = {'file': open(file_path, 'rb')}
        data = additional_data or {}
        
        try:
            response = self.session.post(
                f"{self.base_url}/{endpoint}",
                files=files,
                data=data
            )
            return response.json()
        finally:
            files['file'].close()  # Always close file
    
    def upload_multiple_files(self, endpoint, file_paths):
        """Upload multiple files"""
        files = []
        try:
            for i, file_path in enumerate(file_paths):
                files.append(('files', open(file_path, 'rb')))
            
            response = self.session.post(
                f"{self.base_url}/{endpoint}",
                files=files
            )
            return response.json()
        finally:
            # Close all opened files
            for _, file_obj in files:
                file_obj.close()

# Example usage
client = MultiFormatAPIClient("https://httpbin.org")

# Send JSON
json_response = client.send_json_data("post", {"name": "John", "age": 30})
print("JSON response:", json_response['json'])

# Send form data
form_response = client.send_form_data("post", {"name": "John", "age": "30"})
print("Form response:", form_response['form'])
