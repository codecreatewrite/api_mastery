import requests
import json
from datetime import datetime

# Let's get some fake user data
response = requests.get('https://jsonplaceholder.typicode.com/users')

print(f"Status Code: {response.status_code}")
print(f"Response Headers: {response.headers['content-type']}")
print(f"First user: {response.json()[0]}")
# The requests Response object is powerful
response = requests.get('https://jsonplaceholder.typicode.com/posts/1')

# Different ways to access response data
print("Raw content:", response.content)      # Bytes
print("Text content:", response.text)        # String
print("JSON content:", response.json())      # Python dict/list
print("Status code:", response.status_code)  # Integer
print("Headers:", dict(response.headers))
