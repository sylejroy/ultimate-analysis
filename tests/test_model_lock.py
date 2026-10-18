"""The GPU gate: networks run side by side, an engine set-up runs alone, nobody waits for ever."""

import threading
import time
import unittest

from support import load_module


class GpuGateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.model_lock")
        self.gate = self.module._GpuGate(wait_seconds=0.5)

    def test_networks_run_side_by_side_and_a_set_up_waits_for_them(self):
        began = threading.Event()
        done = threading.Event()
        order = []

        def network():
            with self.gate.work():
                began.set()
                time.sleep(0.1)
                order.append("network")
            done.set()

        thread = threading.Thread(target=network)
        thread.start()
        began.wait(1)
        with self.gate.work():  # A second network does not wait for the first
            order.append("second network")
        with self.gate:
            order.append("set-up")
        thread.join(1)
        self.assertEqual(order, ["second network", "network", "set-up"])

    def test_a_network_run_inside_another_on_one_thread_goes_through(self):
        with self.gate.work():
            with self.gate.work():
                pass
        with self.gate:  # Nobody is left working
            pass

    def test_a_set_up_inside_a_network_run_is_refused_at_once(self):
        with self.gate.work():
            with self.assertRaises(RuntimeError):
                with self.gate:
                    pass

    def test_a_network_that_hangs_holds_up_a_set_up_only_so_long(self):
        release = threading.Event()

        def hanging():
            with self.gate.work():
                release.wait(5)

        thread = threading.Thread(target=hanging)
        thread.start()
        time.sleep(0.05)
        start = time.perf_counter()
        with self.assertRaises(TimeoutError):
            with self.gate:
                pass
        self.assertLess(time.perf_counter() - start, 2.0)
        # The set-up that gave up does not keep others from working
        release.set()
        thread.join(1)
        with self.gate.work():
            pass


if __name__ == "__main__":
    unittest.main()
