# Lesson: city weather comparison tool
Project: api_mastery
Created: 2025-08-29T02:17:02+01:00

## Theory
- 

## Steps / Commands
- 

## Code Highlights
-

## Errors & Fixes
- 

## Next
- 

##Improved code
import os
import requests
from datetime import datetime

API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def fetch_city_weather(city):
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'city': city,
                'weather': data['weather'][0].get('main', 'N/A'),
                'temp': data['main'].get('temp', float('nan')),
                'humidity': data['main'].get('humidity', 'N/A'),
                'description': data['weather'][0].get('description', 'N/A')
            }, None
        else:
            error_msg = response.json().get('message', 'Unknown error')
            return None, f"{response.status_code} - {error_msg}"
    except Exception as e:
        return None, str(e)

def compare_cities_weather(cities):
    # Fetch, display, and process as before...
    pass

if __name__ == "__main__":
    cities = ["London", "New York", "Tokyo", "Sydney", "Warri", "Abraka", "fishy"]
    compare_cities_weather(cities)
