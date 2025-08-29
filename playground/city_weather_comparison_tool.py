#!/usr/bin/env python3
"""
Lesson: city_weather_comparison_tool
Created: 2025-08-29T02:17:02+01:00
Project: api_mastery
Template: script
"""
#Takes multiple city names as input
#Gets weather data for each city
#Displays them in a comparison table
#Shows which city has the highest/lowest temperature

import requests
import json
from datetime import datetime

with open(".secrets.json", newline="") as f:
  API_KEY = json.load(f)['data']['open_weather_api']

#Defaults
#API_KEY = "d58f89640bb0ec79a0a3b5bd2ecd8d7d"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

#Function
# Your challenge template
def compare_cities_weather(cities):
  """
  Compare weather across multiple cities
  Return comparison data
  """
  # Your code here
  # create a list to store data(dict) for each city
  cities_data = []
  skipped_cities = []

  for city in cities:
    params = {
      "q": city,
      "appid": API_KEY,
      "units": "metric"
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
      data = response.json()
    else:
      error = response.status_code
      skipped_cities.append({'city': city, 'error': error})
      continue

    # Collect data eg high_temp, low_temp
    weather = data['weather'][0]['main']
    description = data['weather'][0]['description']
    temp = data['main']['temp']
    humidity = data['main']['humidity']

    #Append collected data to list
    cities_data.append({
      'city': city,
      'weather': weather,
      'temp': temp,
      'humidity': humidity,
      'description': description
    })

  #Extract individual city data from cities data
  print("City Weather Info\n")
  # Print table header
  print(f"{'City':15} {'Weather':10} {'Temp':5} {'Humidity':10} {'Description':20}")
  for city_data in cities_data:
    print(f"{city_data['city']:15} {city_data['weather']:10} {city_data['temp']:5.2f} {city_data['humidity']:>8} {city_data['description']:20}")
  print()

  #Determine city with high & low temp
  if cities_data:
    #cities_data = sorted(cities_data, key=lambda city: (city['temp'], city['humidity']))
    #highest_temp, lowest_temp = cities_data[-1], cities_data[0]
    highest_temp = max(cities_data, key=lambda city: city['temp'])
    lowest_temp = min(cities_data, key=lambda city: city['temp'])
    print(f"{highest_temp['city']} has the highest temperature ({highest_temp['temp']}) overall.")
    print(f"{lowest_temp['city']} has the lowest temperature ({lowest_temp['temp']}) overall.")
  print()

  #Skipped cities
  if skipped_cities:
    print("Skipped Cities")
    for city in skipped_cities:
      print(f"City name: {city['city']} | Error: {city['error']}")

cities = ["London", "New York", "Tokyo", "Sydney", "Warri", "Abraka", "fishy"]
#cities = ["London", "Warri"]
compare_cities_weather(cities)