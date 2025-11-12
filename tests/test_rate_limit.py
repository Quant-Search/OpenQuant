from openquant.utils.rate_limit import RateLimiter


def test_rate_limiter_try_acquire_capacity():
    t = [0.0]

    def time_fn():
        return t[0]

    rl = RateLimiter(rate_per_sec=10.0, capacity=2, time_fn=time_fn)
    assert rl.try_acquire() is True
    assert rl.try_acquire() is True
    # No tokens left immediately
    assert rl.try_acquire() is False
    # Advance 0.1s -> 1 token
    t[0] += 0.1
    assert rl.try_acquire() is True
    # Advance another 0.1s -> 1 token again
    t[0] += 0.1
    assert rl.try_acquire() is True


def test_rate_limiter_blocking_acquire():
    import threading, time

    rl = RateLimiter(rate_per_sec=5.0, capacity=1)
    # Consume initial token
    rl.acquire()

    acquired = []

    def worker():
        rl.acquire()  # should block ~0.2s
        acquired.append(True)

    th = threading.Thread(target=worker)
    th.start()
    th.join(timeout=1.0)
    assert acquired == [True]

