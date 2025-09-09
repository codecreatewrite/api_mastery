#!/usr/bin/env python3
"""
Lesson: rate_limiting_quota_manager
Created: 2025-09-09T19:59:57+01:00
Project: api_mastery
Template: script
"""
import time
import threading
from collections import deque
from typing import Dict, Optional
from datetime import datetime, timedelta

class RateLimiter:
    """
    Intelligent rate limiting for API requests
    """
    
    def __init__(self, calls_per_second: float = 1.0, burst_limit: int = 5):
        self.calls_per_second = calls_per_second
        self.burst_limit = burst_limit
        self.call_times = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the window
            cutoff_time = now - 1.0  # 1 second window
            while self.call_times and self.call_times[0] < cutoff_time:
                self.call_times.popleft()
            
            # Check if we need to wait
            if len(self.call_times) >= self.calls_per_second:
                sleep_time = 1.0 / self.calls_per_second
                time.sleep(sleep_time)
            
            # Record this call
            self.call_times.append(now)
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without waiting"""
        with self.lock:
            now = time.time()
            cutoff_time = now - 1.0
            
            # Count recent calls
            recent_calls = sum(1 for call_time in self.call_times if call_time >= cutoff_time)
            return recent_calls < self.calls_per_second

class QuotaTracker:
    """
    Track daily/monthly API quotas
    """
    
    def __init__(self, daily_limit: Optional[int] = None, monthly_limit: Optional[int] = None):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.daily_count = 0
        self.monthly_count = 0
        self.last_reset_day = datetime.now().date()
        self.last_reset_month = datetime.now().replace(day=1).date()
    
    def check_and_update_quotas(self) -> Dict[str, any]:
        """Check quotas and reset if needed"""
        now = datetime.now()
        today = now.date()
        this_month = now.replace(day=1).date()
        
        # Reset daily counter if needed
        if today > self.last_reset_day:
            self.daily_count = 0
            self.last_reset_day = today
        
        # Reset monthly counter if needed
        if this_month > self.last_reset_month:
            self.monthly_count = 0
            self.last_reset_month = this_month
        
        # Check limits
        daily_ok = self.daily_limit is None or self.daily_count < self.daily_limit
        monthly_ok = self.monthly_limit is None or self.monthly_count < self.monthly_limit
        
        if daily_ok and monthly_ok:
            self.daily_count += 1
            self.monthly_count += 1
        
        return {
            'can_proceed': daily_ok and monthly_ok,
            'daily_remaining': None if self.daily_limit is None else max(0, self.daily_limit - self.daily_count),
            'monthly_remaining': None if self.monthly_limit is None else max(0, self.monthly_limit - self.monthly_count),
            'reason': None if (daily_ok and monthly_ok) else 
                     'Daily limit exceeded' if not daily_ok else 'Monthly limit exceeded'
        }

class SmartAPIClient:
    """
    API client with intelligent rate limiting and quota management
    """
    
    def __init__(self, base_url: str, rate_limit: float = 1.0, daily_limit: int = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(rate_limit)
        self.quota_tracker = QuotaTracker(daily_limit)
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make rate-limited and quota-aware request"""
        
        # Check quota first
        quota_status = self.quota_tracker.check_and_update_quotas()
        if not quota_status['can_proceed']:
            return {
                'success': False,
                'error': quota_status['reason'],
                'quota_status': quota_status
            }
        
        # Wait for rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make the request
        self.request_count += 1
        
        try:
            response = self.session.request(method, f"{self.base_url}/{endpoint}", **kwargs)
            
            # Handle rate limit responses from server
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                self.failed_requests += 1
                
                return {
                    'success': False,
                    'error': 'Rate limited by server',
                    'retry_after': retry_after,
                    'stats': self.get_stats()
                }
            
            elif response.status_code == 200:
                self.successful_requests += 1
                return {
                    'success': True,
                    'data': response.json(),
                    'stats': self.get_stats()
                }
            
            else:
                self.failed_requests += 1
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'stats': self.get_stats()
                }
        
        except requests.RequestException as e:
            self.failed_requests += 1
            return {
                'success': False,
                'error': f'Request failed: {str(e)}',
                'stats': self.get_stats()
            }
    
    def get_stats(self) -> Dict:
        """Get request statistics"""
        success_rate = (self.successful_requests / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            'total_requests': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': f"{success_rate:.1f}%"
        }
    
    def get_quota_status(self) -> Dict:
        """Get current quota status without making a request"""
        return self.quota_tracker.check_and_update_quotas()

if __name__ == '__main__':
    # Example usage
    github_client = SmartAPIClient(
        "https://api.github.com",
        rate_limit=1.0,  # 1 request per second
        daily_limit=5000  # GitHub's limit
    )

    # Make requests that are automatically rate-limited
    for i in range(5):
        result = github_client.make_request("GET", "repos/microsoft/vscode")
        if result['success']:
            print(f"Request {i+1}: Success")
        else:
            print(f"Request {i+1}: {result['error']}")
    
        print(f"Stats: {result['stats']}")
