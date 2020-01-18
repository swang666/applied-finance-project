"""Holds helper code for test functions
"""

import asyncio

def async_test(func):
    """A decorator for async tests

    Args:
        f: the test function

    Returns:
        a decorared function that will run asynchronously in an event_loop()
    """

    def wrapper(*args, **kwargs):
        future = func(*args, **kwargs)
        asyncio.run(future)
    return wrapper
