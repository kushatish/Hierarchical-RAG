from concurrent.futures import ThreadPoolExecutor
import unittest
from core.operations import RateLimiter, RequestMetrics

class OperationsTests(unittest.TestCase):
    def test_rate_limit_boundary_and_expiry(self):
        limiter = RateLimiter(limit=2, window=60)
        self.assertTrue(limiter.allow("a", now=0))
        self.assertTrue(limiter.allow("a", now=1))
        self.assertFalse(limiter.allow("a", now=59))
        self.assertTrue(limiter.allow("a", now=60))
        self.assertTrue(limiter.allow("b", now=60))

    def test_identity_memory_bound(self):
        limiter = RateLimiter(limit=2, identities=2, window=10)
        self.assertTrue(limiter.allow("a", now=0))
        self.assertTrue(limiter.allow("b", now=1))
        self.assertFalse(limiter.allow("c", now=2))
        self.assertTrue(limiter.allow("c", now=11))
        self.assertLessEqual(len(limiter.clients), 2)

    def test_concurrent_rate_limit_cannot_oversubscribe(self):
        limiter = RateLimiter(limit=10)
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(limiter.allow, ["same-client"] * 100))
        self.assertEqual(sum(accepted), 10)

    def test_metrics_count_failures_and_bound_buckets(self):
        metrics = RequestMetrics()
        metrics.record("POST", "/query", 200, 12)
        metrics.record("POST", "/query", 503, 6000)
        report = metrics.snapshot()
        self.assertEqual(report["requests"], 2)
        self.assertEqual(report["latency_non_cumulative_buckets_ms"], {"25": 1, "+Inf": 1})
        self.assertEqual(report["routes"][1]["status"], 503)

class SingleFlightTests(unittest.TestCase):
    def test_duplicate_work_is_shared_and_failures_release_slots(self):
        from core.operations import SingleFlight
        from threading import Barrier, Lock
        import time
        flight = SingleFlight(capacity=1)
        start = Barrier(4)
        calls = []
        lock = Lock()
        def fail():
            with lock:
                calls.append(1)
            time.sleep(.1)
            raise ValueError("provider failed")
        def worker(_):
            start.wait(timeout=5)
            with self.assertRaisesRegex(ValueError, "provider failed"):
                flight.run("same", fail)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, range(4)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(flight.run("retry", lambda: 42), (42, False))

    def test_distinct_request_capacity_is_bounded(self):
        from core.operations import SingleFlight
        from threading import Event
        entered, release = Event(), Event()
        flight = SingleFlight(capacity=1)
        def blocked():
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release leader")
            return 1
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(flight.run, "first", blocked)
            self.assertTrue(entered.wait(timeout=5))
            try:
                with self.assertRaises(RuntimeError):
                    flight.run("second", lambda: 2)
            finally:
                release.set()
            self.assertEqual(task.result(), (1, False))
