#!/usr/bin/env python3
"""
Lesson: weather_app
Created: 2025-08-29T01:44:00+01:00
Project: api_mastery
Template: script
"""
import requests
from datetime import datetime

# Replace with your actual API key
API_KEY = "d58f89640bb0ec79a0a3b5bd2ecd8d7d"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    """
    Get current weather for a city
    This is your first API function!
    """
    # Build the complete URL with parameters
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'  # Celsius instead of Kelvin
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    # Check if request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def display_weather(weather_data, city):
    """
    Display weather data in a readable format
    """
    if not weather_data:
        print(f"Could not get weather data for {city}")
        return

    print(f"\n🌡️  Weather in {city.title()}")
    print("=" * 30)
    print(f"Temperature: {weather_data['main']['temp']}°C")
    print(f"Feels like: {weather_data['main']['feels_like']}°C")
    print(f"Description: {weather_data['weather'][0]['description'].title()}")
    print(f"Humidity: {weather_data['main']['humidity']}%")
    print(f"Pressure: {weather_data['main']['pressure']} hPa")
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Interactive weather checker
while True:
    city = input("\nEnter city name (or 'quit' to exit): ")
    if city.lower() == 'quit':
        break

    weather_data = get_weather(city)
    display_weather(weather_data, city)