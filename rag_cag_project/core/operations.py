"""Bounded, process-local rate limits and request metrics; never store query text."""
from collections import Counter, OrderedDict, deque
from threading import Lock
import time

class RateLimiter:
    def __init__(self, limit=120, window=60, identities=2048):
        if min(limit, window, identities) <= 0:
            raise ValueError("Rate limits must be positive")
        self.limit, self.window, self.identities = limit, window, identities
        self.clients = OrderedDict()
        self.lock = Lock()

    def allow(self, identity, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            while self.clients:
                first, timestamps = next(iter(self.clients.items()))
                if timestamps[-1] > now - self.window:
                    break
                del self.clients[first]
            if identity not in self.clients:
                if len(self.clients) >= self.identities:
                    return False
                self.clients[identity] = deque()
            timestamps = self.clients[identity]
            while timestamps and timestamps[0] <= now - self.window:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            self.clients.move_to_end(identity)
            return True

class RequestMetrics:
    BOUNDS = (5, 10, 25, 50, 100, 250, 500, 1000, 5000)
    def __init__(self):
        self.lock = Lock()
        self.counts = Counter()
        self.histogram = Counter()
        self.total_ms = 0.0

    def record(self, method, route, status, elapsed_ms):
        with self.lock:
            self.counts[(method, route, status)] += 1
            self.total_ms += elapsed_ms
            self.histogram[next((str(bound) for bound in self.BOUNDS if elapsed_ms <= bound), "+Inf")] += 1

    def snapshot(self):
        with self.lock:
            return {"requests": sum(self.counts.values()), "total_latency_ms": round(self.total_ms, 3),
                    "routes": [{"method": m, "route": r, "status": s, "count": n} for (m,r,s),n in sorted(self.counts.items())],
                    "latency_non_cumulative_buckets_ms": dict(self.histogram),
                    "scope": "This worker only; metrics reset on restart"}

class SingleFlight:
    """Share simultaneous identical misses; failures wake all waiters and release slots."""
    def __init__(self, capacity=8):
        if capacity <= 0:
            raise ValueError("In-flight capacity must be positive")
        self.capacity = capacity
        self.pending = {}
        self.lock = Lock()

    def run(self, key, function):
        from concurrent.futures import Future
        with self.lock:
            future = self.pending.get(key)
            shared = future is not None
            if future is None:
                if len(self.pending) >= self.capacity:
                    raise RuntimeError("Too many distinct requests are already generating answers")
                future = Future()
                self.pending[key] = future
        if shared:
            return future.result(timeout=60), True
        try:
            result = function()
            future.set_result(result)
            return result, False
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self.lock:
                self.pending.pop(key, None)
