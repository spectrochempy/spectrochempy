import sys
from time import perf_counter


# ======================================================================================
# context manager to time a block of code
# ======================================================================================
class timeit:
    def __init__(self, msg, test_only=True):
        self.msg = msg
        self.test_only = test_only

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f"Elapsed Time for {self.msg}: {self.time:.6f} seconds\n"
        if not self.test_only or "pytest" in sys.argv[0] or "py.test" in sys.argv[0]:
            print(self.readout)
