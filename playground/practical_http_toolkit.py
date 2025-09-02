#!/usr/bin/env python3
"""
Lesson: practical_http_toolkit
Created: 2025-08-31T19:37:22+01:00
Project: api_mastery
Template: script
"""
#!/usr/bin/env python3
"""
practical_http_toolkit.py

Practical decorator + HTTP client toolkit.

Features:
- timed: measures elapsed time (sync & async)
- retry: retries with exponential backoff and jitter (sync & async)
- rate_limit: token-bucket rate limiter (sync & async)
- ttl_cache: time-to-live cache decorator (sync & async)
- ResilientHTTPClient: a small HTTP client that uses requests (sync)
    and optionally aiohttp (async). Client methods combine the above
    decorators in practical orders (cache -> rate-limit -> retry -> timed).
- Good defaults and thorough inline docs to help you adapt patterns quickly.

Usage:
    pip install requests
    pip install aiohttp   # optional, only required for async client
"""

# ----------------------------
# Standard-library imports
# ----------------------------
import time
import time as _time_module  # alias so comments referencing time.* are clear
import random
import threading
import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, Tuple, Optional

# ----------------------------
# Third-party imports
# ----------------------------
import requests  # sync HTTP client most projects use

# aiohttp is optional; we'll attempt to import and set a flag if available
try:
    import aiohttp  # async HTTP client
    _HAS_AIOHTTP = True
except Exception:
    aiohttp = None  # type: ignore
    _HAS_AIOHTTP = False

# Useful alias for coroutine detection
_is_coroutine = asyncio.iscoroutinefunction


# ============================
# Utility: safe_key for caching
# ============================
def _make_cache_key(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple:
    """
    Make a simple, deterministic key for caching based on args & kwargs.

    IMPORTANT: This is a pragmatic key builder:
      - It uses tuple(args) and a tuple of sorted (k, repr(v)) for kwargs.
      - It will fail or be suboptimal for unhashable complex objects.
      - For production, allow callers to pass a custom key function or use
        a serialization strategy appropriate for your data (JSON, msgpack).
    """
    # Convert kwargs to a sorted tuple of (k, repr(v)) so order doesn't matter
    kwargs_key = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
    return (args, kwargs_key)


# ============================
# Decorator: timed
# ============================
def timed(func: Callable) -> Callable:
    """
    Measure elapsed time for sync and async functions.

    Usage:
        @timed
        def f(...): ...

        @timed
        async def f(...): ...
    """
    if _is_coroutine(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                t1 = time.perf_counter()
                print(f"[timed][async] {func.__name__} -> {t1 - t0:.6f}s")
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                t1 = time.perf_counter()
                print(f"[timed] {func.__ne__} -> {t1 - t0:.6f}s")
        return sync_wrapper


# ============================
# Decorator factory: retry with exponential backoff + jitter
# ============================
def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    jitter: float = 0.1,
    retry_exceptions: Tuple[type, ...] = (Exception,),
):
    """
    Retry decorator factory.

    Backoff formula: delay = min(max_delay, base_delay * backoff_factor ** (attempt - 1))
    Jitter: small random amount added/subtracted to avoid thundering-herd.

    Works for both sync and async functions. For async, uses asyncio.sleep; for sync, time.sleep.

    Example:
        @retry(max_attempts=5, base_delay=0.2)
        def fetch(...): ...
    """
    def decorator(func: Callable) -> Callable:
        if _is_coroutine(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exc = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except retry_exceptions as exc:
                        last_exc = exc
                        # compute backoff delay with jitter
                        delay = min(max_delay, base_delay * (backoff_factor ** (attempt - 1)))
                        # add jitter: uniform(-jitter, jitter) * delay
                        jitter_amount = (random.uniform(-jitter, jitter) * delay)
                        sleep_for = max(0.0, delay + jitter_amount)
                        if attempt >= max_attempts:
                            # re-raise final exception after exhausting attempts
                            raise
                        # wait using asyncio
                        await asyncio.sleep(sleep_for)
                raise last_exc  # unreachable, but keeps type checkers happy
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exc = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except retry_exceptions as exc:
                        last_exc = exc
                        delay = min(max_delay, base_delay * (backoff_factor ** (attempt - 1)))
                        jitter_amount = (random.uniform(-jitter, jitter) * delay)
                        sleep_for = max(0.0, delay + jitter_amount)
                        if attempt >= max_attempts:
                            raise
                        time.sleep(sleep_for)
                raise last_exc
            return sync_wrapper
    return decorator


# ============================
# Decorator factory: TTL cache (in-memory) (sync & async)
# ============================
def ttl_cache(ttl_seconds: float = 60.0, key_builder: Callable = _make_cache_key):
    """
    Simple TTL cache decorator.

    - ttl_seconds: how long an item remains valid (seconds)
    - key_builder: function (args, kwargs) -> hashable key

    Supports both sync and async functions. Thread-safe for sync; uses asyncio.Lock for async.

    Note: caches are kept in memory and will grow unless keys expire. For long-running apps,
    consider using bounded caches, LRU caches, or external caches (Redis, memcached).
    """
    def decorator(func: Callable) -> Callable:
        # storage for cached values: key -> (expiry_ts, value)
        cache: Dict[Any, Tuple[float, Any]] = {}
        lock = threading.Lock()

        if _is_coroutine(func):
            # async variant uses asyncio.Lock for concurrency safety in async context
            a_lock = asyncio.Lock()

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = key_builder(args, kwargs)
                now = time.monotonic()
                # check cache under asyncio lock
                async with a_lock:
                    if key in cache:
                        expiry, value = cache[key]
                        if expiry >= now:
                            # cache hit
                            # NOTE: depending on value mutability you might prefer to return a copy
                            return value
                        else:
                            # expired
                            del cache[key]
                # cache miss -> compute value outside lock
                value = await func(*args, **kwargs)
                expiry = now + ttl_seconds
                async with a_lock:
                    cache[key] = (expiry, value)
                return value

            # expose small API for inspection in interactive sessions
            async_wrapper._cache = cache  # type: ignore
            return async_wrapper

        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                key = key_builder(args, kwargs)
                now = time.monotonic()
                # quick path: check cache under lock
                with lock:
                    entry = cache.get(key)
                    if entry is not None:
                        expiry, value = entry
                        if expiry >= now:
                            return value
                        else:
                            del cache[key]
                # compute without lock
                value = func(*args, **kwargs)
                expiry = now + ttl_seconds
                with lock:
                    cache[key] = (expiry, value)
                return value

            sync_wrapper._cache = cache  # type: ignore
            return sync_wrapper

    return decorator


# ============================
# Decorator factory: rate_limit (token-bucket style)
# ============================
def rate_limit(calls: int, period: float = 1.0):
    """
    Rate limiter decorator factory.

    - calls: allowed number of calls
    - period: time window in seconds

    Token-bucket approach:
    - capacity = calls
    - refill rate = calls / period tokens per second
    - each consume removes 1 token if available; otherwise callers wait (sleep/yield)
    - This decorator blocks the caller until a token is available.
      (For non-blocking behavior, adapt to raise or return a special value.)

    Works for both sync and async functions.
    """
    # tokens are refilled based on monotonic time
    capacity = float(calls)
    refill_rate = float(calls) / float(period)  # tokens per second

    class _SyncBucket:
        def __init__(self):
            self._tokens = capacity
            self._last = time.monotonic()
            self._lock = threading.Lock()

        def consume_blocking(self):
            # Acquire tokens; block using time.sleep until there's at least one token
            while True:
                with self._lock:
                    now = time.monotonic()
                    # refill
                    elapsed = now - self._last
                    if elapsed > 0:
                        self._tokens = min(capacity, self._tokens + elapsed * refill_rate)
                        self._last = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    # need to wait; compute short sleep
                    need = (1.0 - self._tokens) / refill_rate
                # sleep outside lock to allow other threads to proceed
                # sleep a small amount to avoid busy-wait
                time.sleep(min(need, 0.1))

    class _AsyncBucket:
        def __init__(self):
            self._tokens = capacity
            self._last = time.monotonic()
            self._lock = asyncio.Lock()

        async def consume_blocking(self):
            # Async wait until token is available
            while True:
                async with self._lock:
                    now = time.monotonic()
                    elapsed = now - self._last
                    if elapsed > 0:
                        self._tokens = min(capacity, self._tokens + elapsed * refill_rate)
                        self._last = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    need = (1.0 - self._tokens) / refill_rate
                # await a small sleep
                await asyncio.sleep(min(need, 0.1))

    bucket_sync = _SyncBucket()
    bucket_async = _AsyncBucket()

    def decorator(func: Callable) -> Callable:
        if _is_coroutine(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await bucket_async.consume_blocking()
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                bucket_sync.consume_blocking()
                return func(*args, **kwargs)
            return sync_wrapper

    return decorator


# ============================
# ResilientHTTPClient: combines patterns
# ============================
class ResilientHTTPClient:
    """
    A small HTTP client demonstrating practical use of the above decorators.

    Design choices explained inline:
    - For GET requests we typically want caching (so repeated reads are fast).
      We place the cache decorator OUTERMOST so cached calls bypass rate-limiting and retries.
      i.e., @ttl_cache  <-- topmost, so it handles call before inner decorators.
    - Rate limiting sits around network operations to avoid hitting API quota limits.
    - Retry sits around network calls to handle transient network errors.
    - timed is outermost for observability or can be the very outermost depending on wanted measurement.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.3,
        rate_limit_calls: int = 5,
        rate_limit_period: float = 1.0,
        cache_ttl: float = 30.0,
    ):
        # configuration
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Build a requests.Session for sync requests; it's efficient due to connection pooling
        self.session = requests.Session()

        # Optionally configure urllib3 Retry on the session's adapter (better for low-level retries)
        # We keep session-level retry separate; our `retry` decorator handles higher-level retry across the call.
        try:
            from urllib3.util import Retry
            from requests.adapters import HTTPAdapter

            # Keep conservative defaults for adapter-level retry (idempotent methods only)
            adapter_retry = Retry(
                total=max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
                backoff_factor=retry_base_delay,
            )
            adapter = HTTPAdapter(max_retries=adapter_retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        except Exception:
            # urllib3 may still be present but in case of error we proceed without adapter-level retry
            pass

        # Async client: create aiohttp.ClientSession only if aiohttp is installed
        self._async_session = None
        self._async_session_lock = asyncio.Lock()

        # Configure decorators for reuse: we create them here so parameters are centralized
        self._ttl_cache = ttl_cache(ttl_seconds=cache_ttl)
        self._rate_limit = rate_limit(calls=rate_limit_calls, period=rate_limit_period)
        self._retry = retry(
            max_attempts=max_retries,
            base_delay=retry_base_delay,
            backoff_factor=2.0,
            jitter=0.1,
            retry_exceptions=(requests.RequestException, Exception),
        )
        # For async requests we'll pass aiohttp.ClientError to retry_exceptions via separate decorator
        self._async_retry_base = dict(
            max_attempts=max_retries,
            base_delay=retry_base_delay,
            backoff_factor=2.0,
            jitter=0.1,
            max_delay=10.0,
            retry_exceptions=(Exception,)  # we'll refine per-call if needed
        )

    # --------------------
    # Helper: build absolute URL from base_url and path
    # --------------------
    def _build_url(self, path: str) -> str:
        if not self.base_url:
            return path
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    # --------------------
    # Sync GET: decorated in the order we want:
    #   1) Cache outermost -> quick return for repeated identical calls
    #   2) Rate-limit next -> protects API quota when cache missed
    #   3) Retry next -> retry the actual network call on transient errors
    #   4) Timed innermost or outermost depending on whether you want to time only real network ops
    # Here we apply: ttl_cache -> rate_limit -> retry -> timed -> actual network function
    # --------------------
    @ttl_cache  # 1) cache: outermost
    @property  # property so the decorator is evaluated at class creation time; we wrap the method below manually
    def _sync_get_decorators(self):
        """
        We return a composed decorator that applies rate_limit, retry, and timed.
        We use a property to evaluate the composed decorator at runtime with self context.
        """
        # compose decorators: bottom -> top on function definition, runtime order is top-first
        return functools.partial  # placeholder (actual wrapper applied below)

    # To avoid confusion with property above, implement the actual decorated method explicitly:
    def _raw_sync_get(self, path: str, params: dict = None, **kwargs) -> requests.Response:
        """
        The raw network operation using requests. Not decorated.
        """
        url = self._build_url(path)
        # Using session.get is more efficient than requests.get as it uses connection pooling.
        resp = self.session.get(url, params=params, timeout=self.timeout, **kwargs)
        # raise_for_status to surface HTTP errors as exceptions to our retry decorator
        resp.raise_for_status()
        return resp

    # Compose and assign a decorated get method
    # Important: we define it once per-instance (in __init__ would be possible), but for clarity define here:
    def get(self, path: str, params: dict = None, **kwargs) -> requests.Response:
        """
        High-level GET method that applies caching, rate limiting, retry, and timing.

        NOTE on decorator ordering:
          - ttl_cache is applied first (outermost) so cached responses bypass rate limiting &
            retrying (cheap cache lookups).
          - If cache misses, rate_limit ensures we don't exceed API quotas.
          - retry wraps the network call to handle transient failures.
          - timed prints elapsed time for the call that actually executes network operations.
        """
        # Compose decorators in the required order dynamically to properly capture `self` (and its
        # configured decorators like _ttl_cache, _rate_limit, _retry).
        # Step 1: base callable = raw network operation bound to this instance
        base = functools.partial(self._raw_sync_get, path, params, **kwargs)

        # Step 2: apply timed around the raw callable
        base = timed(base)

        # Step 3: apply retry (we need retry to catch requests.RequestException)
        # For the sync retry we already configured retry to include requests.RequestException in __init__
        base = self._retry(base)

        # Step 4: apply rate limit
        base = self._rate_limit(base)

        # Step 5: apply TTL cache (outermost)
        base = self._ttl_cache(base)

        # Now call the fully decorated callable (no arguments because partial applied args)
        return base()

    # Convenience wrapper that returns decoded JSON
    def get_json(self, path: str, params: dict = None, **kwargs) -> Any:
        resp = self.get(path, params=params, **kwargs)
        return resp.json()

    # --------------------
    # Async GET: similar to sync but using aiohttp and async decorators
    # We only enable this if aiohttp is installed. We lazily create a ClientSession for async.
    # --------------------
    async def _ensure_async_session(self):
        if not _HAS_AIOHTTP:
            raise RuntimeError("aiohttp is not installed; async HTTP client is unavailable.")
        async with self._async_session_lock:
            if self._async_session is None or self._async_session.closed:
                # Create a ClientSession; preserve TCPConnector for connection pooling
                self._async_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._async_session

    async def _raw_async_get(self, path: str, params: dict = None, **kwargs) -> Tuple[int, str]:
        """
        Perform async HTTP GET using aiohttp and return (status, text).

        We intentionally return text/content instead of aiohttp.Response to keep cacheability simple.
        """
        session = await self._ensure_async_session()
        url = self._build_url(path)
        async with session.get(url, params=params, **kwargs) as resp:
            text = await resp.text()
            # raise HTTP errors as exceptions similar to requests
            if resp.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info, history=resp.history,
                    status=resp.status, message=resp.reason, headers=resp.headers
                )
            return resp.status, text

    async def async_get(self, path: str, params: dict = None, **kwargs) -> str:
        """
        High-level async GET that uses the async-aware decorators: ttl_cache, rate_limit, retry, timed.

        Behavior mirrors the sync get method.
        """
        if not _HAS_AIOHTTP:
            raise RuntimeError("aiohttp is not installed; async_get unavailable")

        # Build an awaitable partial that calls raw async get
        async def base_call():
            status, text = await self._raw_async_get(path, params=params, **kwargs)
            return text

        # Compose decorators for async path (note: our decorators are async-aware)
        decorated = base_call
        # timed supports coroutine functions directly
        decorated = timed(decorated)
        # async retry: create a decorator with proper retry_exceptions that include aiohttp exceptions
        retry_async = retry(**{**self._async_retry_base, "retry_exceptions": (Exception,)})
        decorated = retry_async(decorated)
        # async rate limiter
        decorated = rate_limit(calls=5, period=1.0)(decorated)
        # async TTL cache
        decorated = ttl_cache(ttl_seconds=30.0)(decorated)

        # Now call
        return await decorated()

    async def async_close(self):
        """
        Close the aiohttp session when you're done (good practice).
        """
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()


# ----------------------------
# Example usage & small demo
# ----------------------------
def demo_sync_usage():
    """
    Demo: use ResilientHTTPClient to fetch a JSON placeholder post a couple times
    to demonstrate caching, timing, rate-limiting, and retry.
    """
    client = ResilientHTTPClient(
        base_url="https://jsonplaceholder.typicode.com",
        timeout=5.0,
        max_retries=3,
        retry_base_delay=0.2,
        rate_limit_calls=2,  # small rate limit to demo throttling
        rate_limit_period=1.0,
        cache_ttl=10.0,
    )

    # First call: cache miss -> network call occurs (timed/retried/rate-limited)
    print("First call (expected network):")
    post = client.get_json("/posts/1")
    print("title:", post.get("title"))

    # Second call: cached -> should be fast and avoid hitting rate limit or retry
    print("\nSecond call (should be cached):")
    post2 = client.get_json("/posts/1")
    print("title:", post2.get("title"))

    # Rapidly call a different path to demonstrate rate limiting
    print("\nRapid calls to show rate limiting (some will wait):")
    for i in range(5):
        try:
            client.get_json("/posts/2")
            print(f"fetch {i+1} OK")
        except Exception as e:
            print("fetch failed:", e)


async def demo_async_usage():
    """
    Demo async usage if aiohttp is installed.
    """
    if not _HAS_AIOHTTP:
        print("aiohttp not installed; skipping async demo. Install with: pip install aiohttp")
        return

    client = ResilientHTTPClient(base_url="https://jsonplaceholder.typicode.com", rate_limit_calls=2, rate_limit_period=1.0)
    try:
        print("Async first call (network):")
        data = await client.async_get("/posts/1")
        print("len:", len(data))
        print("\nAsync second call (cached):")
        data2 = await client.async_get("/posts/1")
        print("len:", len(data2))
    finally:
        await client.async_close()


# ----------------------------
# CLI / quick-run demo
# ----------------------------
if __name__ == "__main__":
    print("Practical HTTP toolkit demo (sync).")
    demo_sync_usage()

    # run async demo if aiohttp available
    if _HAS_AIOHTTP:
        print("\nPractical HTTP toolkit demo (async).")
        asyncio.run(demo_async_usage())
    else:
        print("\nAsync demo skipped: aiohttp not installed. (pip install aiohttp)")

# ----------------------------
# End of script
# ----------------------------
