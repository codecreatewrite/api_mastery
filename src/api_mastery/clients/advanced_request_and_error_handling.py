# Module 4: Advanced Requests & Error Handling
# Your API Mastery Journey - Days 12-15

import asyncio
import aiohttp
import requests
import json
import time
import logging
import statistics
import threading
import queue
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps

# Part 1: Asynchronous Requests - Breaking the Speed Barrier (75 minutes)

@dataclass
class RequestConfig:
    """Configuration for async requests"""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    max_concurrent: int = 10

class AsyncAPIClient:
    """Professional async API client with connection pooling and advanced features"""
    
    def __init__(self, base_url: str, headers: Dict[str, str] = None, config: RequestConfig = None):
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.config = config or RequestConfig()
        
        # Connection management
        self.connector = None
        self.session = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retries_used': 0,
            'total_time': 0.0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _create_session(self):
        """Create aiohttp session with optimized settings"""
        # Configure connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=10,
            sock_read=self.config.timeout
        )
        
        self.connector = connector
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers,
            json_serialize=json.dumps
        )
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def _make_request_with_retry(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make request with automatic retry logic"""
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    # Handle different status codes
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'success': True,
                            'data': data,
                            'status': response.status,
                            'headers': dict(response.headers),
                            'attempt': attempt + 1
                        }
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(retry_after)
                            continue
                    elif 500 <= response.status < 600:  # Server error
                        if attempt < self.config.max_retries:
                            delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                            await asyncio.sleep(delay)
                            continue
                    
                    # Non-retryable error or max retries exceeded
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f'HTTP {response.status}',
                        'status': response.status,
                        'details': error_text[:200],
                        'attempt': attempt + 1
                    }
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries:
                    self.stats['retries_used'] += 1
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    return {
                        'success': False,
                        'error': 'Request timeout',
                        'attempt': attempt + 1
                    }
            except Exception as e:
                if attempt < self.config.max_retries:
                    self.stats['retries_used'] += 1
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'attempt': attempt + 1
                    }
        
        return {
            'success': False,
            'error': 'Max retries exceeded',
            'attempt': self.config.max_retries + 1
        }
    
    async def get(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Async GET request"""
        async with self.semaphore:  # Limit concurrent requests
            start_time = asyncio.get_event_loop().time()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            result = await self._make_request_with_retry('GET', url, params=params)
            
            # Update statistics
            self.stats['total_requests'] += 1
            if result['success']:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            elapsed = asyncio.get_event_loop().time() - start_time
            self.stats['total_time'] += elapsed
            result['elapsed_time'] = elapsed
            
            return result
    
    async def post(self, endpoint: str, data: Dict = None, json_data: Dict = None) -> Dict[str, Any]:
        """Async POST request"""
        async with self.semaphore:
            start_time = asyncio.get_event_loop().time()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            kwargs = {}
            if json_data:
                kwargs['json'] = json_data
            elif data:
                kwargs['data'] = data
                
            result = await self._make_request_with_retry('POST', url, **kwargs)
            
            # Update statistics
            self.stats['total_requests'] += 1
            if result['success']:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            elapsed = asyncio.get_event_loop().time() - start_time
            self.stats['total_time'] += elapsed
            result['elapsed_time'] = elapsed
            
            return result
    
    async def batch_get(self, endpoints: List[str], params_list: List[Dict] = None) -> List[Dict[str, Any]]:
        """Make multiple GET requests concurrently"""
        if params_list is None:
            params_list = [None] * len(endpoints)
            
        tasks = [
            self.get(endpoint, params)
            for endpoint, params in zip(endpoints, params_list)
        ]
        
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
            
        return {
            **self.stats,
            'success_rate': (self.stats['successful_requests'] / total) * 100,
            'average_time': self.stats['total_time'] / total,
            'requests_per_second': total / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
        }

class AsyncWeatherDashboard:
    """High-performance async weather dashboard"""
    
    def __init__(self, openweather_key: str, weatherapi_key: str = None):
        self.openweather_key = openweather_key
        self.weatherapi_key = weatherapi_key
        
        self.openweather_client = AsyncAPIClient("http://api.openweathermap.org/data/2.5")
        self.weatherapi_client = AsyncAPIClient("http://api.weatherapi.com/v1") if weatherapi_key else None
    
    async def get_weather_data(self, city: str, sources: List[str] = None) -> Dict:
        """Get weather data from multiple sources concurrently"""
        if sources is None:
            sources = ['openweather']
            if self.weatherapi_client:
                sources.append('weatherapi')
        
        tasks = []
        
        if 'openweather' in sources:
            task = self._get_openweather_data(city)
            tasks.append(('openweather', task))
            
        if 'weatherapi' in sources and self.weatherapi_client:
            task = self._get_weatherapi_data(city)
            tasks.append(('weatherapi', task))
            
        # Execute all requests concurrently
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (source, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    results[source] = {'success': False, 'error': str(result)}
                else:
                    results[source] = result
                    
        return results
    
    async def _get_openweather_data(self, city: str) -> Dict:
        """Get OpenWeatherMap data asynchronously"""
        params = {
            'q': city,
            'appid': self.openweather_key,
            'units': 'metric'
        }
        
        result = await self.openweather_client.get('weather', params)
        
        if result['success']:
            raw_data = result['data']
            return {
                'success': True,
                'source': 'OpenWeatherMap',
                'data': {
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
            }
        else:
            return {'success': False, 'source': 'OpenWeatherMap', 'error': result['error']}
    
    async def _get_weatherapi_data(self, city: str) -> Dict:
        """Get WeatherAPI data asynchronously"""
        params = {
            'key': self.weatherapi_key,
            'q': city,
            'aqi': 'no'
        }
        
        result = await self.weatherapi_client.get('current.json', params)
        
        if result['success']:
            raw_data = result['data']
            return {
                'success': True,
                'source': 'WeatherAPI',
                'data': {
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
            }
        else:
            return {'success': False, 'source': 'WeatherAPI', 'error': result['error']}
    
    async def get_multiple_cities_weather(self, cities: List[str]) -> Dict[str, Dict]:
        """Get weather for multiple cities concurrently"""
        tasks = {city: self.get_weather_data(city) for city in cities}
        results = await asyncio.gather(*tasks.values())
        
        return dict(zip(cities, results))
    
    async def close(self):
        """Clean up resources"""
        await self.openweather_client.close()
        if self.weatherapi_client:
            await self.weatherapi_client.close()

# Part 2: Streaming Responses and Real-Time Data (60 minutes)

class StreamingClient:
    """Handle streaming responses efficiently"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def stream_large_file(self, endpoint: str, chunk_size: int = 8192) -> Generator[bytes, None, None]:
        """Stream large file download"""
        url = f"{self.base_url}/{endpoint}"
        
        with self.session.get(url, stream=True) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    downloaded += len(chunk)
                    
                    # Progress reporting
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r⬇️ Download progress: {progress:.1f}%", end='', flush=True)
                    
                    yield chunk
            
            print()  # New line after progress
    
    def stream_json_lines(self, endpoint: str) -> Generator[Dict, None, None]:
        """Stream JSON Lines format (one JSON object per line)"""
        url = f"{self.base_url}/{endpoint}"
        
        with self.session.get(url, stream=True) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line.decode('utf-8'))
                        yield json_obj
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Invalid JSON line: {line.decode('utf-8')[:50]}...")
                        continue
    
    def stream_server_sent_events(self, endpoint: str, on_event: Callable[[Dict], None] = None) -> Generator[Dict, None, None]:
        """Stream Server-Sent Events (SSE)"""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache'
        }
        
        with self.session.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()
            
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    buffer += chunk
                    
                    # Process complete events
                    while '\n\n' in buffer:
                        event_data, buffer = buffer.split('\n\n', 1)
                        event = self._parse_sse_event(event_data)
                        
                        if event:
                            if on_event:
                                on_event(event)
                            yield event
    
    def _parse_sse_event(self, event_data: str) -> Dict:
        """Parse Server-Sent Event data"""
        event = {
            'event': None,
            'data': '',
            'id': None,
            'retry': None
        }
        
        for line in event_data.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'data':
                    event['data'] += value + '\n'
                else:
                    event[key] = value
        
        # Clean up data field
        event['data'] = event['data'].rstrip('\n')
        
        # Parse JSON data if possible
        if event['data']:
            try:
                event['parsed_data'] = json.loads(event['data'])
            except json.JSONDecodeError:
                event['parsed_data'] = event['data']
        
        return event

class AsyncStreamingClient:
    """Async streaming for better performance"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    async def stream_large_file_async(self, endpoint: str, chunk_size: int = 8192):
        """Async streaming file download"""
        url = f"{self.base_url}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                async for chunk in response.content.iter_chunked(chunk_size):
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r⬇️ Download progress: {progress:.1f}%", end='', flush=True)
                    
                    yield chunk
                
                print()
    
    async def stream_json_lines_async(self, endpoint: str):
        """Async JSON Lines streaming"""
        url = f"{self.base_url}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    try:
                        json_obj = json.loads(line.decode('utf-8'))
                        yield json_obj
                    except json.JSONDecodeError:
                        continue

class RealTimeDataFeed:
    """Simulate real-time data streaming (like Twitter, stock prices, etc.)"""
    
    def __init__(self):
        self.is_running = False
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[Dict], None]):
        """Add callback for processing streaming data"""
        self.callbacks.append(callback)
    
    async def start_feed(self, feed_type: str = "social"):
        """Start streaming real-time data"""
        self.is_running = True
        print(f"🔴 Starting {feed_type} data stream...")
        
        while self.is_running:
            # Simulate real-time data
            data = self._generate_fake_data(feed_type)
            
            # Process with all callbacks
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    print(f"⚠️ Callback error: {e}")
            
            # Wait before next data point
            await asyncio.sleep(1)
    
    def stop_feed(self):
        """Stop streaming"""
        self.is_running = False
        print("⏹️ Data stream stopped")
    
    def _generate_fake_data(self, feed_type: str) -> Dict:
        """Generate fake streaming data"""
        import random
        
        if feed_type == "social":
            return {
                'type': 'social_post',
                'user': f"user_{random.randint(1, 1000)}",
                'content': f"This is post #{random.randint(1, 10000)}",
                'likes': random.randint(0, 100),
                'timestamp': datetime.now().isoformat(),
                'engagement_score': random.uniform(0, 1)
            }
        elif feed_type == "stock":
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            symbol = random.choice(symbols)
            
            return {
                'type': 'stock_price',
                'symbol': symbol,
                'price': round(random.uniform(100, 1000), 2),
                'change': round(random.uniform(-5, 5), 2),
                'volume': random.randint(1000, 100000),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'type': 'generic',
                'data': f"Generic data point {random.randint(1, 1000)}",
                'value': random.uniform(0, 100),
                'timestamp': datetime.now().isoformat()
            }

class StreamingDashboard:
    """Dashboard that processes streaming data in real-time"""
    
    def __init__(self):
        self.data_buffer = []
        self.stats = {
            'total_messages': 0,
            'messages_per_second': 0,
            'last_update': datetime.now()
        }
        
        self.feed = RealTimeDataFeed()
        self.feed.add_callback(self.process_data)
    
    async def process_data(self, data: Dict):
        """Process incoming streaming data"""
        self.data_buffer.append(data)
        self.stats['total_messages'] += 1
        
        # Keep only last 100 messages
        if len(self.data_buffer) > 100:
            self.data_buffer = self.data_buffer[-100:]
        
        # Update stats every 10 messages
        if self.stats['total_messages'] % 10 == 0:
            await self.update_dashboard()
    
    async def update_dashboard(self):
        """Update dashboard display"""
        now = datetime.now()
        elapsed = (now - self.stats['last_update']).total_seconds()
        
        if elapsed > 0:
            self.stats['messages_per_second'] = 10 / elapsed
            self.stats['last_update'] = now
        
        # Display current stats
        print(f"\r📊 Messages: {self.stats['total_messages']} | "
              f"Rate: {self.stats['messages_per_second']:.1f}/sec | "
              f"Buffer: {len(self.data_buffer)}", end='', flush=True)
    
    async def start_dashboard(self, feed_type: str = "social", duration: int = 30):
        """Start streaming dashboard"""
        print(f"🚀 Starting streaming dashboard ({duration} seconds)")
        
        # Start the data feed
        feed_task = asyncio.create_task(self.feed.start_feed(feed_type))
        
        # Run for specified duration
        await asyncio.sleep(duration)
        
        # Stop the feed
        self.feed.stop_feed()
        feed_task.cancel()
        
        try:
            await feed_task
        except asyncio.CancelledError:
            pass
        
        print(f"\n✅ Dashboard completed. Processed {self.stats['total_messages']} messages")
        
        # Show sample of recent data
        if self.data_buffer:
            print("\n📄 Recent messages:")
            for i, message in enumerate(self.data_buffer[-5:], 1):
                print(f"  {i}. {message['type']}: {str(message)[:60]}...")

# Part 3: Circuit Breakers and Advanced Retry Strategies (75 minutes)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Professional circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 expected_exception: tuple = (Exception,), success_threshold: int = 3):
        # Configuration
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_opens': 0,
            'circuit_closes': 0,
            'calls_prevented': 0
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.stats['total_calls'] += 1
        
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                print(f"🔄 Circuit breaker transitioning to HALF_OPEN")
            else:
                self.stats['calls_prevented'] += 1
                raise Exception("Circuit breaker is OPEN - call prevented")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success handling
            self._on_success()
            return result
            
        except self.expected_exception as e:
            # Failure handling
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.stats['successful_calls'] += 1
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close_circuit()
    
    def _on_failure(self):
        """Handle failed call"""
        self.stats['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit"""
        self.state = CircuitState.OPEN
        self.stats['circuit_opens'] += 1
        print(f"🔴 Circuit breaker OPENED after {self.failure_count} failures")
    
    def _close_circuit(self):
        """Close the circuit"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.stats['circuit_closes'] += 1
        print(f"✅ Circuit breaker CLOSED - service recovered")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try resetting"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'stats': self.stats.copy(),
            'last_failure_time': self.last_failure_time
        }

class SmartRetryStrategy:
    """Advanced retry strategy with multiple algorithms"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, strategy: str = "exponential_backoff"):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        
        # Response time tracking for adaptive strategies
        self.response_times = []
        self.success_rates = []
    
    def calculate_delay(self, attempt: int, last_response_time: float = None) -> float:
        """Calculate delay based on strategy"""
        if self.strategy == "fixed":
            return self.base_delay
            
        elif self.strategy == "linear":
            return min(self.base_delay * attempt, self.max_delay)
            
        elif self.strategy == "exponential_backoff":
            delay = self.base_delay * (2 ** (attempt - 1))
            return min(delay, self.max_delay)
            
        elif self.strategy == "exponential_backoff_jitter":
            import random
            delay = self.base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0.1, 0.9)
            return min(delay * jitter, self.max_delay)
            
        elif self.strategy == "adaptive":
            # Adapt based on recent response times
            if self.response_times:
                avg_response_time = statistics.mean(self.response_times[-10:])
                delay = self.base_delay * (avg_response_time / 1.0)  # Normalize to 1 second
                return min(delay * (2 ** (attempt - 1)), self.max_delay)
            else:
                return self.calculate_delay(attempt, None)  # Fall back to exponential
                
        else:
            return self.base_delay
    
    def should_retry(self, attempt: int, exception: Exception, response_code: int = None) -> bool:
        """Decide if we should retry based on the error"""
        if attempt >= self.max_retries:
            return False
        
        # Don't retry client errors (4xx), except for specific cases
        if response_code and 400 <= response_code < 500:
            # Retry these specific client errors
            retryable_4xx = [408, 429]  # Timeout, Rate Limited
            return response_code in retryable_4xx
        
        # Retry server errors (5xx)
        if response_code and 500 <= response_code < 600:
            return True
        
        # Retry network-related errors
        retryable_exceptions = [
            'ConnectionError',
            'Timeout',
            'ConnectTimeout',
            'ReadTimeout'
        ]
        
        exception_name = type(exception).__name__
        return exception_name in retryable_exceptions
    
    def record_response_time(self, response_time: float):
        """Record response time for adaptive strategies"""
        self.response_times.append(response_time)
        
        # Keep only recent measurements
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-50:]

class ResilientAPIClient:
    """Ultimate resilient API client with circuit breaker and smart retries"""
    
    def __init__(self, base_url: str, circuit_breaker: CircuitBreaker = None, 
                 retry_strategy: SmartRetryStrategy = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Initialize circuit breaker
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30
        )
        
        # Initialize retry strategy
        self.retry_strategy = retry_strategy or SmartRetryStrategy(
            max_retries=3,
            strategy="exponential_backoff_jitter"
        )
        
        # Performance tracking
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'circuit_breaker_blocks': 0,
            'retries_used': 0,
            'average_response_time': 0,
            'response_times': []
        }
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make resilient API request"""
        
        async def _execute_request():
            """Inner function to execute the actual request"""
            start_time = time.time()
            
            try:
                url = f"{self.base_url}/{endpoint.lstrip('/')}"
                response = self.session.request(method, url, timeout=30, **kwargs)
                response_time = time.time() - start_time
                
                self.retry_strategy.record_response_time(response_time)
                
                # Update metrics
                self.metrics['response_times'].append(response_time)
                if len(self.metrics['response_times']) > 100:
                    self.metrics['response_times'] = self.metrics['response_times'][-50:]
                
                self.metrics['average_response_time'] = statistics.mean(self.metrics['response_times'])
                
                if response.status_code == 200:
                    return {
                        'success': True,
                        'data': response.json(),
                        'status_code': response.status_code,
                        'response_time': response_time
                    }
                else:
                    # Raise exception for non-200 status codes
                    response.raise_for_status()
                    
            except Exception as e:
                response_time = time.time() - start_time
                raise e
        
        # Execute with circuit breaker and retry logic
        for attempt in range(1, self.retry_strategy.max_retries + 2):
            try:
                self.metrics['total_requests'] += 1
                
                # Use circuit breaker
                result = await self.circuit_breaker.call(_execute_request)
                self.metrics['successful_requests'] += 1
                
                return result
                
            except Exception as e:
                self.metrics['failed_requests'] += 1
                
                # Get response code if available
                response_code = getattr(e, 'response', {}).get('status_code') if hasattr(e, 'response') else None
                
                # Check if we should retry
                if self.retry_strategy.should_retry(attempt, e, response_code):
                    delay = self.retry_strategy.calculate_delay(attempt)
                    self.metrics['retries_used'] += 1
                    
                    print(f"⏳ Attempt {attempt} failed, retrying in {delay:.2f}s: {str(e)[:50]}...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # No more retries or non-retryable error
                    return {
                        'success': False,
                        'error': str(e),
                        'attempt': attempt,
                        'circuit_breaker_state': self.circuit_breaker.get_state()
                    }
        
        # Should never reach here
        return {
            'success': False,
            'error': 'Maximum retries exceeded',
            'circuit_breaker_state': self.circuit_breaker.get_state()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive client metrics"""
        cb_state = self.circuit_breaker.get_state()
        
        return {
            'request_metrics': self.metrics,
            'circuit_breaker': cb_state,
            'success_rate': (
                self.metrics['successful_requests'] / self.metrics['total_requests'] * 100
                if self.metrics['total_requests'] > 0 else 0
            )
        }

class UnreliableAPISimulator:
    """Simulate an unreliable API for testing circuit breaker and retry logic"""
    
    def __init__(self, failure_rate: float = 0.3, response_delay: float = 1.0):
        self.failure_rate = failure_rate
        self.response_delay = response_delay
        self.call_count = 0
    
    async def make_call(self) -> Dict[str, Any]:
        """Simulate API call with potential failures"""
        import random
        
        self.call_count += 1
        
        # Simulate network delay
        await asyncio.sleep(self.response_delay)
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            # Randomly choose failure type
            failure_types = [
                ('ConnectionError', 'Connection failed'),
                ('Timeout', 'Request timeout'),
                ('ServerError', 'Internal server error')
            ]
            
            failure_type, message = random.choice(failure_types)
            raise Exception(f"{failure_type}: {message}")
        
        # Success
        return {
            'success': True,
            'data': {
                'message': f'Success on call #{self.call_count}',
                'timestamp': datetime.now().isoformat()
            }
        }

# Part 4: Middleware and Request/Response Hooks (60 minutes)

class RequestMiddleware:
    """Base class for request middleware"""
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request before sending"""
        return request_data
    
    def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process response after receiving"""
        return response_data
    
    def process_error(self, error: Exception, request_data: Dict[str, Any]) -> Exception:
        """Process errors"""
        return error

class LoggingMiddleware(RequestMiddleware):
    """Comprehensive request/response logging"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(f"api_client.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log outgoing requests"""
        self.logger.info(
            f"🚀 REQUEST: {request_data['method']} {request_data['url']} "
            f"(Headers: {len(request_data.get('headers', {}))}, "
            f"Body: {len(str(request_data.get('data', '')))} chars)"
        )
        
        # Log headers (excluding sensitive ones)
        headers = request_data.get('headers', {})
        safe_headers = {
            k: v if k.lower() not in ['authorization', 'x-api-key'] else '***HIDDEN***'
            for k, v in headers.items()
        }
        
        self.logger.debug(f"Request headers: {safe_headers}")
        return request_data
    
    def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log incoming responses"""
        status = response_data.get('status_code', 'Unknown')
        elapsed = response_data.get('elapsed_time', 0)
        size = len(str(response_data.get('data', '')))
        
        self.logger.info(
            f"📥 RESPONSE: {status} in {elapsed:.3f}s ({size} chars)"
        )
        
        return response_data
    
    def process_error(self, error: Exception, request_data: Dict[str, Any]) -> Exception:
        """Log errors"""
        self.logger.error(
            f"❌ ERROR: {request_data['method']} {request_data['url']} - {str(error)}"
        )
        
        return error

class MetricsMiddleware(RequestMiddleware):
    """Collect detailed performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0.0,
            'response_times': [],
            'status_codes': {},
            'errors': {},
            'endpoints': {}
        }
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track request start"""
        request_data['_start_time'] = time.time()
        self.metrics['total_requests'] += 1
        return request_data
    
    def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track response metrics"""
        # Calculate response time
        start_time = response_data.get('_start_time', time.time())
        elapsed = time.time() - start_time
        response_data['elapsed_time'] = elapsed
        
        # Update metrics
        self.metrics['successful_requests'] += 1
        self.metrics['total_time'] += elapsed
        self.metrics['response_times'].append(elapsed)
        
        # Track status codes
        status = response_data.get('status_code', 'Unknown')
        self.metrics['status_codes'][status] = self.metrics['status_codes'].get(status, 0) + 1
        
        # Track endpoints
        url = response_data.get('url', 'Unknown')
        self.metrics['endpoints'][url] = self.metrics['endpoints'].get(url, 0) + 1
        
        # Keep only recent response times
        if len(self.metrics['response_times']) > 1000:
            self.metrics['response_times'] = self.metrics['response_times'][-500:]
        
        return response_data
    
    def process_error(self, error: Exception, request_data: Dict[str, Any]) -> Exception:
        """Track error metrics"""
        self.metrics['failed_requests'] += 1
        
        error_type = type(error).__name__
        self.metrics['errors'][error_type] = self.metrics['errors'].get(error_type, 0) + 1
        
        return error
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        import statistics
        
        total = self.metrics['total_requests']
        if total == 0:
            return self.metrics
        
        response_times = self.metrics['response_times']
        
        return {
            'total_requests': total,
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': (self.metrics['successful_requests'] / total) * 100,
            'average_response_time': statistics.mean(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            'requests_per_second': total / self.metrics['total_time'] if self.metrics['total_time'] > 0 else 0,
            'most_common_status': max(self.metrics['status_codes'].items(), key=lambda x: x[1]) if self.metrics['status_codes'] else None,
            'most_common_error': max(self.metrics['errors'].items(), key=lambda x: x[1]) if self.metrics['errors'] else None,
            'top_endpoints': sorted(self.metrics['endpoints'].items(), key=lambda x: x[1], reverse=True)[:5]
        }

class CachingMiddleware(RequestMiddleware):
    """Intelligent response caching"""
    
    def __init__(self, ttl_seconds: int = 300, max_cache_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_times = {}
    
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key from request data"""
        method = request_data['method']
        url = request_data['url']
        params = request_data.get('params', {})
        
        # Only cache GET requests
        if method.upper() != 'GET':
            return None
        
        # Create key from URL and parameters
        cache_key = f"{method}:{url}"
        if params:
            sorted_params = sorted(params.items())
            cache_key += f"?{','.join(f'{k}={v}' for k, v in sorted_params)}"
        
        return cache_key
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached item is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]['timestamp']
        return (time.time() - cached_time) < self.ttl_seconds
    
    def _cleanup_cache(self):
        """Remove expired and excess cache entries"""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, value in self.cache.items()
            if (current_time - value['timestamp']) >= self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
        
        # Remove excess entries (LRU)
        if len(self.cache) > self.max_cache_size:
            # Sort by access time and remove oldest
            sorted_by_access = sorted(self.access_times.items(), key=lambda x: x[1])
            excess_count = len(self.cache) - self.max_cache_size
            
            for key, _ in sorted_by_access[:excess_count]:
                if key in self.cache:
                    del self.cache[key]
                    del self.access_times[key]
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check cache before making request"""
        cache_key = self._generate_cache_key(request_data)
        
        if cache_key and self._is_cache_valid(cache_key):
            # Return cached response
            cached_data = self.cache[cache_key]['data']
            self.access_times[cache_key] = time.time()
            
            # Mark as cached
            cached_data['_from_cache'] = True
            cached_data['_cache_age'] = time.time() - self.cache[cache_key]['timestamp']
            
            # Skip actual request by raising special exception
            raise CacheHitException(cached_data)
        
        request_data['_cache_key'] = cache_key
        return request_data
    
    def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cache successful responses"""
        cache_key = response_data.get('_cache_key')
        
        if cache_key and response_data.get('status_code') == 200:
            # Cache the response
            self.cache[cache_key] = {
                'data': response_data.copy(),
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
            
            # Cleanup if needed
            self._cleanup_cache()
        
        # Remove internal cache key
        if '_cache_key' in response_data:
            del response_data['_cache_key']
        
        return response_data

class CacheHitException(Exception):
    """Special exception to indicate cache hit"""
    
    def __init__(self, cached_data):
        self.cached_data = cached_data
        super().__init__("Cache hit")

class RateLimitingMiddleware(RequestMiddleware):
    """Client-side rate limiting"""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rate limiting before request"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        return request_data

class MiddlewareAPIClient:
    """API client with extensible middleware system"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.middleware_stack: List[RequestMiddleware] = []
    
    def add_middleware(self, middleware: RequestMiddleware):
        """Add middleware to the stack"""
        self.middleware_stack.append(middleware)
    
    def remove_middleware(self, middleware_type: type):
        """Remove middleware of specific type"""
        self.middleware_stack = [
            m for m in self.middleware_stack
            if not isinstance(m, middleware_type)
        ]
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make request through middleware stack"""
        # Prepare request data
        request_data = {
            'method': method.upper(),
            'url': f"{self.base_url}/{endpoint.lstrip('/')}",
            'headers': kwargs.get('headers', {}),
            'params': kwargs.get('params', {}),
            'data': kwargs.get('data'),
            'json': kwargs.get('json')
        }
        
        try:
            # Process request through middleware stack
            for middleware in self.middleware_stack:
                request_data = middleware.process_request(request_data)
            
            # Make actual HTTP request
            response = self.session.request(
                request_data['method'],
                request_data['url'],
                headers=request_data['headers'],
                params=request_data['params'],
                data=request_data['data'],
                json=request_data['json'],
                timeout=30
            )
            
            # Prepare response data
            response_data = {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                'url': response.url,
                '_start_time': request_data.get('_start_time')
            }
            
            # Process response through middleware stack (in reverse order)
            for middleware in reversed(self.middleware_stack):
                response_data = middleware.process_response(response_data)
            
            return response_data
            
        except CacheHitException as e:
            # Handle cache hit
            return e.cached_data
            
        except Exception as e:
            # Process error through middleware stack
            for middleware in self.middleware_stack:
                e = middleware.process_error(e, request_data)
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'request_data': request_data
            }

# Example: Building a production-ready API client

class ProductionAPIClient(MiddlewareAPIClient):
    """Production-ready API client with all middleware configured"""
    
    def __init__(self, base_url: str, api_key: str = None):
        super().__init__(base_url)
        
        # Add authentication header if API key provided
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        # Set up middleware stack
        self.setup_middleware()
    
    def setup_middleware(self):
        """Configure production middleware stack"""
        # 1. Rate limiting (respect API limits)
        self.add_middleware(RateLimitingMiddleware(requests_per_second=2.0))
        
        # 2. Caching (reduce unnecessary requests)
        self.add_middleware(CachingMiddleware(ttl_seconds=300))
        
        # 3. Metrics collection
        self.metrics_middleware = MetricsMiddleware()
        self.add_middleware(self.metrics_middleware)
        
        # 4. Comprehensive logging
        self.add_middleware(LoggingMiddleware(log_level="INFO"))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return self.metrics_middleware.get_metrics_summary()

# Part 5: Production-Grade Error Recovery and Monitoring (75 minutes)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Comprehensive error context"""
    timestamp: datetime
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    endpoint: str
    method: str
    request_id: str
    user_context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: str = None

class IntelligentErrorClassifier:
    """Classify errors intelligently for appropriate recovery strategies"""
    
    def __init__(self):
        self.error_patterns = {
            ErrorCategory.NETWORK: [
                'ConnectionError',
                'ConnectTimeout',
                'ReadTimeout',
                'DNSLookupError',
                'NetworkError'
            ],
            ErrorCategory.AUTHENTICATION: [
                'Unauthorized',
                'Invalid API key',
                'Token expired',
                'Authentication failed'
            ],
            ErrorCategory.AUTHORIZATION: [
                'Forbidden',
                'Access denied',
                'Insufficient permissions'
            ],
            ErrorCategory.RATE_LIMIT: [
                'Rate limit',
                'Too many requests',
                'Quota exceeded',
                'Throttled'
            ],
            ErrorCategory.SERVER_ERROR: [
                'Internal server error',
                'Service unavailable',
                'Bad gateway',
                'Gateway timeout'
            ],
            ErrorCategory.TIMEOUT: [
                'Timeout',
                'Request timeout',
                'Read timeout'
            ]
        }
        
        self.severity_rules = {
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.AUTHENTICATION: ErrorSeverity.HIGH,
            ErrorCategory.AUTHORIZATION: ErrorSeverity.HIGH,
            ErrorCategory.RATE_LIMIT: ErrorSeverity.MEDIUM,
            ErrorCategory.SERVER_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.CLIENT_ERROR: ErrorSeverity.LOW,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM
        }
    
    def classify_error(self, error: Exception, response_code: int = None) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error into category and severity"""
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Check response codes first
        if response_code:
            if response_code == 401:
                return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
            elif response_code == 403:
                return ErrorCategory.AUTHORIZATION, ErrorSeverity.HIGH
            elif response_code == 429:
                return ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM
            elif 500 <= response_code < 600:
                return ErrorCategory.SERVER_ERROR, ErrorSeverity.HIGH
            elif 400 <= response_code < 500:
                return ErrorCategory.CLIENT_ERROR, ErrorSeverity.LOW
        
        # Check error patterns
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_message or pattern.lower() in error_type.lower():
                    return category, self.severity_rules[category]
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

class AdaptiveRecoveryEngine:
    """Intelligent error recovery with learning capabilities"""
    
    def __init__(self):
        self.recovery_strategies = {
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.RATE_LIMIT: self._recover_rate_limit,
            ErrorCategory.SERVER_ERROR: self._recover_server_error,
            ErrorCategory.TIMEOUT: self._recover_timeout,
            ErrorCategory.AUTHENTICATION: self._recover_authentication
        }
        
        # Learning system - track what works
        self.recovery_success_rates = {}
        self.recovery_history = []
    
    async def attempt_recovery(self, error_context: ErrorContext, recovery_callback: Callable = None) -> bool:
        """Attempt intelligent error recovery"""
        strategy_name = f"{error_context.category.value}_recovery"
        
        if error_context.category in self.recovery_strategies:
            recovery_func = self.recovery_strategies[error_context.category]
            
            try:
                print(f"🔄 Attempting recovery for {error_context.category.value} error...")
                success = await recovery_func(error_context, recovery_callback)
                
                # Record recovery attempt
                self._record_recovery_attempt(error_context.category, strategy_name, success)
                
                error_context.recovery_attempted = True
                error_context.recovery_successful = success
                error_context.recovery_strategy = strategy_name
                
                if success:
                    print(f"✅ Recovery successful using {strategy_name}")
                else:
                    print(f"❌ Recovery failed for {strategy_name}")
                
                return success
                
            except Exception as e:
                print(f"⚠️ Recovery strategy failed: {str(e)}")
                return False
        
        return False
    
    async def _recover_network_error(self, error_context: ErrorContext, callback: Callable = None) -> bool:
        """Recover from network errors"""
        # Strategy: Progressive backoff with connectivity check
        backoff_delays = [1, 2, 5, 10]
        
        for delay in backoff_delays:
            print(f"  Waiting {delay}s before retry...")
            await asyncio.sleep(delay)
            
            # Simple connectivity check
            if await self._check_connectivity():
                print("  Network connectivity restored")
                return True
        
        return False
    
    async def _recover_rate_limit(self, error_context: ErrorContext, callback: Callable = None) -> bool:
        """Recover from rate limiting"""
        # Strategy: Respect Retry-After header or use exponential backoff
        retry_after = 60  # Default fallback
        
        # In a real implementation, you'd extract this from response headers
        print(f"  Rate limited - waiting {retry_after}s...")
        await asyncio.sleep(retry_after)
        
        return True
    
    async def _recover_server_error(self, error_context: ErrorContext, callback: Callable = None) -> bool:
        """Recover from server errors"""
        # Strategy: Short delays with health check
        for attempt in range(3):
            delay = 2 ** attempt  # 1, 2, 4 seconds
            await asyncio.sleep(delay)
            
            # In production, you might ping a health endpoint
            if await self._check_server_health(error_context.endpoint):
                return True
        
        return False
    
    async def _recover_timeout(self, error_context: ErrorContext, callback: Callable = None) -> bool:
        """Recover from timeout errors"""
        # Strategy: Increase timeout and retry
        print("  Timeout recovery - increasing timeout duration")
        await asyncio.sleep(1)
        
        return True
    
    async def _recover_authentication(self, error_context: ErrorContext, callback: Callable = None) -> bool:
        """Recover from authentication errors"""
        # Strategy: Attempt token refresh
        print("  Attempting token refresh...")
        
        if callback:
            # Call token refresh callback
            return await callback() if asyncio.iscoroutinefunction(callback) else callback()
        
        return False
    
    async def _check_connectivity(self) -> bool:
        """Simple connectivity check"""
        # In production, ping a reliable endpoint
        await asyncio.sleep(0.1)  # Simulate check
        return True
    
    async def _check_server_health(self, endpoint: str) -> bool:
        """Check if server is healthy"""
        # In production, call health endpoint
        await asyncio.sleep(0.1)  # Simulate check
        return True
    
    def _record_recovery_attempt(self, category: ErrorCategory, strategy: str, success: bool):
        """Record recovery attempt for learning"""
        key = f"{category.value}_{strategy}"
        
        if key not in self.recovery_success_rates:
            self.recovery_success_rates[key] = {'attempts': 0, 'successes': 0}
        
        self.recovery_success_rates[key]['attempts'] += 1
        
        if success:
            self.recovery_success_rates[key]['successes'] += 1
        
        # Keep recent history
        self.recovery_history.append({
            'timestamp': datetime.now(),
            'category': category.value,
            'strategy': strategy,
            'success': success
        })
        
        # Keep only recent history
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-500:]
    
    def get_recovery_analytics(self) -> Dict[str, Any]:
        """Get recovery performance analytics"""
        analytics = {
            'strategy_performance': {},
            'recent_recovery_rate': 0,
            'most_effective_strategy': None,
            'least_effective_strategy': None
        }
        
        # Calculate success rates for each strategy
        for key, data in self.recovery_success_rates.items():
            if data['attempts'] > 0:
                success_rate = (data['successes'] / data['attempts']) * 100
                analytics['strategy_performance'][key] = {
                    'success_rate': success_rate,
                    'attempts': data['attempts'],
                    'successes': data['successes']
                }
        
        # Find most/least effective strategies
        if analytics['strategy_performance']:
            sorted_strategies = sorted(
                analytics['strategy_performance'].items(),
                key=lambda x: x[1]['success_rate'],
                reverse=True
            )
            
            if sorted_strategies:
                analytics['most_effective_strategy'] = sorted_strategies[0]
                analytics['least_effective_strategy'] = sorted_strategies[-1]
        
        # Recent recovery rate (last 100 attempts)
        recent_attempts = self.recovery_history[-100:] if self.recovery_history else []
        if recent_attempts:
            recent_successes = sum(1 for attempt in recent_attempts if attempt['success'])
            analytics['recent_recovery_rate'] = (recent_successes / len(recent_attempts)) * 100
        
        return analytics

class RealTimeErrorMonitor:
    """Real-time error monitoring and alerting system"""
    
    def __init__(self, alert_threshold: int = 5, time_window_minutes: int = 5):
        self.alert_threshold = alert_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        self.error_buffer = []
        self.alert_callbacks = []
        
        # Thread-safe queue for error processing
        self.error_queue = queue.Queue()
        self.monitor_thread = None
        self.is_monitoring = False
    
    def start_monitoring(self):
        """Start real-time error monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_errors, daemon=True)
            self.monitor_thread.start()
            print("🔍 Error monitoring started")
    
    def stop_monitoring(self):
        """Stop error monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏹️ Error monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[List[ErrorContext]], None]):
        """Add callback for error alerts"""
        self.alert_callbacks.append(callback)
    
    def record_error(self, error_context: ErrorContext):
        """Record an error for monitoring"""
        self.error_queue.put(error_context)
    
    def _monitor_errors(self):
        """Background thread to monitor errors"""
        while self.is_monitoring:
            try:
                # Process errors from queue (with timeout)
                error_context = self.error_queue.get(timeout=1.0)
                self._process_error(error_context)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Error in monitor thread: {e}")
    
    def _process_error(self, error_context: ErrorContext):
        """Process individual error"""
        current_time = datetime.now()
        
        # Add to buffer
        self.error_buffer.append(error_context)
        
        # Remove old errors outside time window
        cutoff_time = current_time - self.time_window
        self.error_buffer = [
            err for err in self.error_buffer
            if err.timestamp > cutoff_time
        ]
        
        # Check for alert conditions
        self._check_alert_conditions()
    
    def _check_alert_conditions(self):
        """Check if alert should be triggered"""
        # Simple threshold alert
        if len(self.error_buffer) >= self.alert_threshold:
            self._trigger_alert("threshold_exceeded", self.error_buffer.copy())
        
        # High severity errors
        high_severity_errors = [
            err for err in self.error_buffer
            if err.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        ]
        
        if len(high_severity_errors) >= 2:
            self._trigger_alert("high_severity_cluster", high_severity_errors)
        
        # Same error category clustering
        category_counts = {}
        for error in self.error_buffer:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1
        
        for category, count in category_counts.items():
            if count >= 3:
                category_errors = [err for err in self.error_buffer if err.category == category]
                self._trigger_alert("category_cluster", category_errors, {"category": category.value})
    
    def _trigger_alert(self, alert_type: str, errors: List[ErrorContext], additional_data: Dict = None):
        """Trigger error alert"""
        alert_data = {
            'alert_type': alert_type,
            'timestamp': datetime.now(),
            'error_count': len(errors),
            'errors': errors,
            'additional_data': additional_data or {}
        }
        
        print(f"🚨 ALERT: {alert_type} - {len(errors)} errors detected")
        
        # Call all alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                print(f"⚠️ Alert callback failed: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get current error summary"""
        current_time = datetime.now()
        cutoff_time = current_time - self.time_window
        
        recent_errors = [
            err for err in self.error_buffer
            if err.timestamp > cutoff_time
        ]
        
        if not recent_errors:
            return {'total_errors': 0, 'time_window_minutes': self.time_window.total_seconds() / 60}
        
        # Analyze errors
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'time_window_minutes': self.time_window.total_seconds() / 60,
            'errors_by_category': category_counts,
            'errors_by_severity': severity_counts,
            'most_common_category': max(category_counts.items(), key=lambda x: x[1]) if category_counts else None,
            'critical_errors': severity_counts.get('critical', 0),
            'high_severity_errors': severity_counts.get('high', 0)
        }

# Ultimate Production API Client

class UltimateAPIClient:
    """Production-grade API client with comprehensive error handling and monitoring"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        # Initialize systems
        self.error_classifier = IntelligentErrorClassifier()
        self.recovery_engine = AdaptiveRecoveryEngine()
        self.error_monitor = RealTimeErrorMonitor()
        
        # Setup monitoring and alerting
        self.error_monitor.add_alert_callback(self._handle_error_alert)
        self.error_monitor.start_monitoring()
        
        # Request tracking
        self.request_counter = 0
        self.active_requests = {}
    
    async def make_resilient_request(self, method: str, endpoint: str, max_recovery_attempts: int = 2, **kwargs) -> Dict[str, Any]:
        """Make request with comprehensive error handling and recovery"""
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        self.active_requests[request_id] = {
            'method': method,
            'endpoint': endpoint,
            'start_time': datetime.now(),
            'attempts': 0
        }
        
        recovery_attempts = 0
        
        while recovery_attempts <= max_recovery_attempts:
            try:
                self.active_requests[request_id]['attempts'] += 1
                
                # Make the request
                url = f"{self.base_url}/{endpoint.lstrip('/')}"
                response = self.session.request(method, url, timeout=30, **kwargs)
                
                # Success
                del self.active_requests[request_id]
                
                return {
                    'success': True,
                    'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                    'status_code': response.status_code,
                    'request_id': request_id,
                    'recovery_attempts': recovery_attempts
                }
                
            except Exception as e:
                # Classify the error
                response_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                category, severity = self.error_classifier.classify_error(e, response_code)
                
                # Create error context
                error_context = ErrorContext(
                    timestamp=datetime.now(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    category=category,
                    severity=severity,
                    endpoint=endpoint,
                    method=method,
                    request_id=request_id,
                    user_context={'recovery_attempt': recovery_attempts}
                )
                
                # Record error for monitoring
                self.error_monitor.record_error(error_context)
                
                # Attempt recovery if not at max attempts and severity is not low
                if recovery_attempts < max_recovery_attempts and severity != ErrorSeverity.LOW:
                    print(f"🔄 Attempting recovery {recovery_attempts + 1}/{max_recovery_attempts} for {request_id}")
                    
                    recovery_success = await self.recovery_engine.attempt_recovery(
                        error_context,
                        recovery_callback=self._token_refresh_callback
                    )
                    
                    if recovery_success:
                        recovery_attempts += 1
                        continue
                
                # Recovery failed or not attempted
                del self.active_requests[request_id]
                
                return {
                    'success': False,
                    'error': str(e),
                    'error_category': category.value,
                    'error_severity': severity.value,
                    'request_id': request_id,
                    'recovery_attempts': recovery_attempts,
                    'recovery_attempted': error_context.recovery_attempted,
                    'recovery_successful': error_context.recovery_successful
                }
        
        # Should not reach here
        return {
            'success': False,
            'error': 'Maximum recovery attempts exceeded',
            'request_id': request_id
        }
    
    async def _token_refresh_callback(self) -> bool:
        """Callback for token refresh during authentication recovery"""
        # In production, implement actual token refresh logic
        print("  Simulating token refresh...")
        await asyncio.sleep(0.1)
        return True
    
    def _handle_error_alert(self, alert_data: Dict[str, Any]):
        """Handle error alerts"""
        alert_type = alert_data['alert_type']
        error_count = alert_data['error_count']
        
        print(f"\n🚨 ERROR ALERT: {alert_type.upper()}")
        print(f"  Time: {alert_data['timestamp']}")
        print(f"  Error count: {error_count}")
        
        if alert_data['additional_data']:
            print(f"  Additional data: {alert_data['additional_data']}")
        
        # In production, send to monitoring system, Slack, email, etc.
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive client status"""
        return {
            'active_requests': len(self.active_requests),
            'total_requests': self.request_counter,
            'error_summary': self.error_monitor.get_error_summary(),
            'recovery_analytics': self.recovery_engine.get_recovery_analytics(),
            'longest_running_request': self._get_longest_running_request()
        }
    
    def _get_longest_running_request(self) -> Optional[Dict[str, Any]]:
        """Get longest running request info"""
        if not self.active_requests:
            return None
        
        longest = max(
            self.active_requests.items(),
            key=lambda x: x[1]['start_time']
        )
        
        request_id, request_data = longest
        duration = (datetime.now() - request_data['start_time']).total_seconds()
        
        return {
            'request_id': request_id,
            'duration_seconds': duration,
            'endpoint': request_data['endpoint'],
            'attempts': request_data['attempts']
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.error_monitor.stop_monitoring()

# Demo functions

def demo_middleware_system():
    """Demonstrate the middleware system"""
    
    print("🔧 Testing Middleware System")
    print("=" * 40)
    
    # Create client with all middleware
    client = ProductionAPIClient("https://jsonplaceholder.typicode.com")
    
    print("📊 Making test requests...")
    
    # Make several requests to demonstrate middleware
    endpoints = ['posts/1', 'posts/2', 'posts/1', 'users/1', 'posts/3']
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n🔄 Request {i}: GET /{endpoint}")
        result = client.make_request('GET', endpoint)
        
        if result.get('_from_cache'):
            print(f"  💨 Served from cache (age: {result['_cache_age']:.1f}s)")
        elif result.get('success', True):
            print(f"  ✅ Success: {result['status_code']}")
        else:
            print(f"  ❌ Error: {result['error']}")
        
        time.sleep(0.5)  # Small delay
    
    # Show performance report
    print(f"\n📈 Performance Report:")
    report = client.get_performance_report()
    
    print(f"  Total requests: {report['total_requests']}")
    print(f"  Success rate: {report['success_rate']:.1f}%")
    print(f"  Average response time: {report['average_response_time']:.3f}s")
    print(f"  Requests per second: {report['requests_per_second']:.1f}")
    
    if report['top_endpoints']:
        print(f"  Most accessed endpoint: {report['top_endpoints'][0][0]} ({report['top_endpoints'][0][1]} times)")

async def demo_ultimate_client():
    """Demonstrate the ultimate production API client"""
    
    print("🚀 Ultimate Production API Client Demo")
    print("=" * 50)
    
    client = UltimateAPIClient("https://jsonplaceholder.typicode.com")
    
    try:
        # Make several requests to demonstrate features
        test_requests = [
            ('GET', 'posts/1'),
            ('GET', 'posts/999999'),  # Will cause 404
            ('GET', 'posts/2'),
            ('POST', 'posts', {'title': 'Test', 'body': 'Test body', 'userId': 1})
        ]
        
        for i, (method, endpoint, *args) in enumerate(test_requests, 1):
            print(f"\n🔄 Request {i}: {method} /{endpoint}")
            
            kwargs = {}
            if args:
                kwargs['json'] = args[0]
            
            result = await client.make_resilient_request(method, endpoint, **kwargs)
            
            if result['success']:
                print(f"  ✅ Success: {result['status_code']}")
                if result.get('recovery_attempts', 0) > 0:
                    print(f"  🔄 Required {result['recovery_attempts']} recovery attempts")
            else:
                print(f"  ❌ Failed: {result['error']}")
                print(f"  📊 Category: {result.get('error_category', 'unknown')}")
                print(f"  ⚠️ Severity: {result.get('error_severity', 'unknown')}")
        
        # Wait a moment for error processing
        await asyncio.sleep(2)
        
        # Show comprehensive status
        print(f"\n📊 Client Status:")
        status = client.get_comprehensive_status()
        
        print(f"  Total requests: {status['total_requests']}")
        print(f"  Active requests: {status['active_requests']}")
        
        error_summary = status['error_summary']
        print(f"  Recent errors: {error_summary['total_errors']}")
        
        if error_summary['most_common_category']:
            category, count = error_summary['most_common_category']
            print(f"  Most common error: {category} ({count} times)")
        
        recovery_analytics = status['recovery_analytics']
        if recovery_analytics['most_effective_strategy']:
            strategy, data = recovery_analytics['most_effective_strategy']
            print(f"  Best recovery strategy: {strategy} ({data['success_rate']:.1f}% success)")
    
    finally:
        client.cleanup()

# Performance comparison

def compare_performance():
    """Compare sequential vs async performance"""
    
    # ❌ SLOW: Sequential requests (blocking)
    def slow_multiple_requests(urls: List[str]) -> List[Dict]:
        """Traditional blocking approach - slow for multiple requests"""
        results = []
        start_time = time.time()
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                results.append({
                    'url': url,
                    'status': response.status_code,
                    'data': response.json() if response.status_code == 200 else None,
                    'error': None
                })
            except Exception as e:
                results.append({
                    'url': url,
                    'status': None,
                    'data': None,
                    'error': str(e)
                })
        
        end_time = time.time()
        print(f"Sequential requests took: {end_time - start_time:.2f} seconds")
        return results
    
    # ✅ FAST: Asynchronous requests (non-blocking)
    async def fast_multiple_requests(urls: List[str]) -> List[Dict]:
        """Async approach - much faster for multiple requests"""
        start_time = time.time()
        
        async def fetch_one(session: aiohttp.ClientSession, url: str) -> Dict:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json() if response.status == 200 else None
                    return {
                        'url': url,
                        'status': response.status,
                        'data': data,
                        'error': None
                    }
            except Exception as e:
                return {
                    'url': url,
                    'status': None,
                    'data': None,
                    'error': str(e)
                }
        
        # Create session and make all requests concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_one(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        print(f"Async requests took: {end_time - start_time:.2f} seconds")
        return results
    
    test_urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
        "https://jsonplaceholder.typicode.com/posts/3",
        "https://jsonplaceholder.typicode.com/posts/4",
        "https://jsonplaceholder.typicode.com/posts/5"
    ]
    
    print("🐌 Testing sequential requests...")
    sequential_results = slow_multiple_requests(test_urls)
    
    print("\n⚡ Testing async requests...")
    async_results = asyncio.run(fast_multiple_requests(test_urls))
    
    print(f"\nResults: {len(sequential_results)} vs {len(async_results)} successful requests")

async def demo_async_client():
    """Demonstrate advanced async API client usage"""
    
    config = RequestConfig(
        timeout=15,
        max_retries=2,
        max_concurrent=5
    )
    
    async with AsyncAPIClient("https://jsonplaceholder.typicode.com", config=config) as client:
        # Single request
        print("🔍 Making single request...")
        result = await client.get("posts/1")
        
        if result['success']:
            print(f"✅ Post title: {result['data']['title']}")
        
        # Batch requests
        print("\n🚀 Making batch requests...")
        endpoints = [f"posts/{i}" for i in range(1, 11)]
        batch_results = await client.batch_get(endpoints)
        
        successful = sum(1 for r in batch_results if r['success'])
        print(f"✅ Batch completed: {successful}/{len(batch_results)} successful")
        
        # Show statistics
        stats = client.get_stats()
        print(f"\n📊 Performance Stats:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Average time: {stats['average_time']:.3f}s")
        print(f"  Requests/second: {stats['requests_per_second']:.1f}")

async def demo_async_weather():
    """Demonstrate async weather dashboard"""
    
    # Replace with your actual API keys
    OPENWEATHER_KEY = "your_openweather_api_key"
    
    dashboard = AsyncWeatherDashboard(OPENWEATHER_KEY)
    
    try:
        # Get weather for multiple cities simultaneously
        cities = ["London", "New York", "Tokyo", "Sydney", "Paris"]
        print(f"🌍 Getting weather for {len(cities)} cities simultaneously...")
        
        start_time = asyncio.get_event_loop().time()
        results = await dashboard.get_multiple_cities_weather(cities)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        print(f"⚡ Completed in {elapsed:.2f} seconds")
        
        # Display results
        for city, data in results.items():
            if 'openweather' in data and data['openweather']['success']:
                weather = data['openweather']['data']
                print(f"\n🌡️ {city}: {weather['temperature']}°C, {weather['description']}")
            else:
                print(f"\n❌ {city}: Failed to get weather data")
    
    finally:
        await dashboard.close()

async def demo_streaming():
    """Demonstrate streaming capabilities"""
    
    dashboard = StreamingDashboard()
    print("🌊 Testing streaming dashboard...")
    await dashboard.start_dashboard("social", duration=10)

async def demo_resilient_client():
    """Demonstrate circuit breaker and retry strategies"""
    
    print("🛡️ Testing Resilient API Client")
    print("=" * 40)
    
    # Create circuit breaker with aggressive settings for demo
    circuit_breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5,
        success_threshold=2
    )
    
    # Create retry strategy
    retry_strategy = SmartRetryStrategy(
        max_retries=3,
        strategy="exponential_backoff_jitter"
    )
    
    # Create resilient client
    client = ResilientAPIClient(
        "https://httpbin.org",
        circuit_breaker=circuit_breaker,
        retry_strategy=retry_strategy
    )
    
    # Simulate unreliable service
    unreliable_service = UnreliableAPISimulator(failure_rate=0.6, response_delay=0.5)
    
    # Test calls
    for i in range(15):
        try:
            print(f"\n🔄 Making request #{i+1}...")
            
            # Simulate request with circuit breaker
            result = await circuit_breaker.call(unreliable_service.make_call)
            print(f"✅ Success: {result['data']['message']}")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
        
        # Show circuit breaker state
        cb_state = circuit_breaker.get_state()
        print(f"  Circuit: {cb_state['state']} | Failures: {cb_state['failure_count']}")
        
        # Small delay between requests
        await asyncio.sleep(1)
    
    # Show final statistics
    print(f"\n📊 Final Statistics:")
    cb_stats = circuit_breaker.stats
    
    print(f"  Total calls: {cb_stats['total_calls']}")
    print(f"  Successful: {cb_stats['successful_calls']}")
    print(f"  Failed: {cb_stats['failed_calls']}")
    print(f"  Calls prevented: {cb_stats['calls_prevented']}")
    print(f"  Circuit opens: {cb_stats['circuit_opens']}")
    print(f"  Circuit closes: {cb_stats['circuit_closes']}")

