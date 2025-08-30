#!/usr/bin/env python3
"""
Lesson: json_placeholder_crud_operations_m2
Created: 2025-08-30T20:23:22+01:00
Project: api_mastery
Template: script
"""
import requests
import json

BASE_URL = "https://jsonplaceholder.typicode.com"

class BlogAPI:
    """
    Your first API client class!
    We'll build on this throughout the module
    """

    def __init__(self):
        self.base_url = BASE_URL

    def get_post(self, post_id):
        """READ operation - Get a specific post"""
        response = requests.get(f"{self.base_url}/posts/{post_id}")
        return self._handle_response(response)

    def create_post(self, title, body, user_id):
        """CREATE operation - Add new post"""
        data = {
            'title': title,
            'body': body,
            'userId': user_id
        }

        response = requests.post(
            f"{self.base_url}/posts",
            json=data,  # Automatically sets Content-Type header
            headers={'Content-Type': 'application/json'}
        )
        return self._handle_response(response)

    def update_post(self, post_id, title, body, user_id):
        """UPDATE operation - Replace entire post"""
        data = {
            'id': post_id,
            'title': title,
            'body': body,
            'userId': user_id
        }

        response = requests.put(
            f"{self.base_url}/posts/{post_id}",
            json=data
        )
        return self._handle_response(response)

    def patch_post(self, post_id, **updates):
        """PATCH operation - Partial update"""
        response = requests.patch(
            f"{self.base_url}/posts/{post_id}",
            json=updates
        )
        return self._handle_response(response)

    def delete_post(self, post_id):
        """DELETE operation - Remove post"""
        response = requests.delete(f"{self.base_url}/posts/{post_id}")
        return self._handle_response(response)

    def _handle_response(self, response):
        """Private method to handle all responses consistently"""
        if response.status_code in [200, 201]:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

# Test your API client
blog = BlogAPI()

# CREATE
new_post = blog.create_post(
    title="My API Learning Journey",
    body="I'm mastering APIs step by step!",
    user_id=1
)
print("Created post:", new_post)

# READ
post = blog.get_post(1)
print("Retrieved post:", post['title'])

# UPDATE (partial)
updated = blog.patch_post(1, title="Updated: My API Journey")
print("Updated post:", updated)

# DELETE
blog.delete_post(1)
print("Post deleted!")