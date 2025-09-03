#!/usr/bin/env python3
"""
Lesson: advanced_weather_app_m2
Created: 2025-09-03T23:35:18+01:00
Project: api_mastery
Template: script
"""
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from robustapiclient import RobustAPIClient

class WeatherDashboard:
    """
    Professional weather dashboard with multiple APIs and caching
    """

    def __init__(self, openweather_key: str, weatherapi_key: str = None):
        self.openweather_key = openweather_key
        self.weatherapi_key = weatherapi_key

        # Initialize robust API clients
        self.openweather_client = RobustAPIClient("http://api.openweathermap.org/data/2.5")
        self.weatherapi_client = RobustAPIClient("http://api.weatherapi.com/v1") if weatherapi_key else None

        # Simple in-memory cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

    def _get_cache_key(self, city: str, provider: str) -> str:
        """Generate cache key"""
        return f"{provider}_{city.lower()}"

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cached data is still valid"""
        cache_time = cache_entry.get('timestamp', 0)
        return time.time() - cache_time < self.cache_duration

    def _cache_data(self, key: str, data: Dict) -> None:
        """Store data in cache with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }

    def get_openweather_data(self, city: str) -> Optional[Dict]:
        """Get weather data from OpenWeatherMap"""
        cache_key = self._get_cache_key(city, 'openweather')

        # Check cache first
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"📋 Using cached OpenWeather data for {city}")
            return self.cache[cache_key]['data']

        # Make API request
        params = {
            'q': city,
            'appid': self.openweather_key,
            'units': 'metric'
        }

        result = self.openweather_client.get('weather', params=params)

        if result['success']:
            # Transform data to common format
            raw_data = result['data']
            weather_data = {
                'provider': 'OpenWeatherMap',
                'city': raw_data['name'],
                'country': raw_data['sys']['country'],
                'temperature': raw_data['main']['temp'],
                'feels_like': raw_data['main']['feels_like'],
                'description': raw_data['weather'][0]['description'],
                'humidity': raw_data['main']['humidity'],
                'pressure': raw_data['main']['pressure'],
                'wind_speed': raw_data.get('wind', {}).get('speed', 'N/A'),
                'timestamp': datetime.now().isoformat()
            }

            # Cache the result
            self._cache_data(cache_key, weather_data)
            return weather_data

        else:
            print(f"❌ OpenWeather error: {result['error']}")
            return None

    def get_weatherapi_data(self, city: str) -> Optional[Dict]:
        """Get weather data from WeatherAPI (alternative source)"""
        if not self.weatherapi_client:
            return None

        cache_key = self._get_cache_key(city, 'weatherapi')

        # Check cache first
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"📋 Using cached WeatherAPI data for {city}")
            return self.cache[cache_key]['data']

        params = {
            'key': self.weatherapi_key,
            'q': city,
            'aqi': 'no'
        }

        result = self.weatherapi_client.get('current.json', params=params)

        if result['success']:
            raw_data = result['data']
            weather_data = {
                'provider': 'WeatherAPI',
                'city': raw_data['location']['name'],
                'country': raw_data['location']['country'],
                'temperature': raw_data['current']['temp_c'],
                'feels_like': raw_data['current']['feelslike_c'],
                'description': raw_data['current']['condition']['text'],
                'humidity': raw_data['current']['humidity'],
                'pressure': raw_data['current']['pressure_mb'],
                'wind_speed': raw_data['current']['wind_kph'],
                'timestamp': datetime.now().isoformat()
            }

            self._cache_data(cache_key, weather_data)
            return weather_data

        else:
            print(f"❌ WeatherAPI error: {result['error']}")
            return None

    def get_weather_comparison(self, city: str) -> Dict:
        """Get weather from multiple sources and compare"""
        results = {}

        # Try OpenWeatherMap
        openweather_data = self.get_openweather_data(city)
        if openweather_data:
            results['openweather'] = openweather_data

        # Try WeatherAPI if available
        if self.weatherapi_client:
            weatherapi_data = self.get_weatherapi_data(city)
            if weatherapi_data:
                results['weatherapi'] = weatherapi_data

        return results

    def display_weather_comparison(self, city: str) -> None:
        """Display weather data from multiple sources"""
        print(f"\n🌤️  Weather Comparison for {city.title()}")
        print("=" * 50)

        comparison = self.get_weather_comparison(city)

        if not comparison:
            print("❌ No weather data available")
            return

        for provider, data in comparison.items():
            print(f"\n📊 {data['provider']} Data:")
            print(f"   Temperature: {data['temperature']}°C (feels like {data['feels_like']}°C)")
            print(f"   Description: {data['description'].title()}")
            print(f"   Humidity: {data['humidity']}%")
            print(f"   Pressure: {data['pressure']} hPa")
            print(f"   Wind Speed: {data['wind_speed']} m/s")
            print(f"   Updated: {data['timestamp']}")

        # Show temperature difference if multiple sources
        if len(comparison) > 1:
            temps = [data['temperature'] for data in comparison.values()]
            temp_diff = max(temps) - min(temps)
            print(f"\n📈 Temperature difference between sources: {temp_diff:.1f}°C")

    def get_multiple_cities(self, cities: List[str]) -> Dict:
        """Get weather for multiple cities efficiently"""
        results = {}

        for city in cities:
            print(f"🔍 Fetching weather for {city}...")
            weather_data = self.get_openweather_data(city)
            if weather_data:
                results[city] = weather_data

            # Small delay to be respectful to API
            time.sleep(0.1)

        return results

    def display_cities_comparison(self, cities: List[str]) -> None:
        """Display weather comparison across multiple cities"""
        results = self.get_multiple_cities(cities)

        if not results:
            print("❌ No weather data retrieved")
            return

        print(f"\n🗺️  Multi-City Weather Report")
        print("=" * 60)

        # Sort cities by temperature
        sorted_cities = sorted(results.items(), key=lambda x: x[1]['temperature'], reverse=True)

        for city, data in sorted_cities:
            print(f"{city.title():15} | {data['temperature']:5.1f}°C | {data['description'].title()}")

        # Statistics
        temps = [data['temperature'] for data in results.values()]
        print(f"\n📊 Temperature Statistics:")
        print(f"   Highest: {max(temps):.1f}°C")
        print(f"   Lowest: {min(temps):.1f}°C")
        print(f"   Average: {sum(temps)/len(temps):.1f}°C")

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        print("🗑️  Cache cleared")

    def show_cache_status(self):
        """Show current cache status"""
        print(f"📋 Cache contains {len(self.cache)} entries")
        for key, entry in self.cache.items():
            age = time.time() - entry['timestamp']
            print(f"   {key}: {age:.0f} seconds old")

# Enhanced Weather Dashboard Usage
def main():
    # Replace with your actual API keys
    OPENWEATHER_KEY = "d58f89640bb0ec79a0a3b5bd2ecd8d7d"
    WEATHERAPI_KEY = None  # Optional second API

    dashboard = WeatherDashboard(OPENWEATHER_KEY, WEATHERAPI_KEY)

    while True:
        print("\n🌡️  Enhanced Weather Dashboard")
        print("1. Single city weather")
        print("2. Compare multiple cities")
        print("3. Weather comparison (multiple sources)")
        print("4. Show cache status")
        print("5. Clear cache")
        print("6. Quit")

        choice = input("\nSelect option (1-6): ")

        if choice == '1':
            city = input("Enter city name: ")
            weather_data = dashboard.get_openweather_data(city)
            if weather_data:
                dashboard.display_weather_comparison(city)

        elif choice == '2':
            cities_input = input("Enter cities separated by commas: ")
            cities = [city.strip() for city in cities_input.split(',')]
            dashboard.display_cities_comparison(cities)

        elif choice == '3':
            city = input("Enter city name: ")
            dashboard.display_weather_comparison(city)

        elif choice == '4':
            dashboard.show_cache_status()

        elif choice == '5':
            dashboard.clear_cache()

        elif choice == '6':
            print("👋 Thanks for using the Weather Dashboard!")
            break

        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()
