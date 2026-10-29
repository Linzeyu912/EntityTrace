import io
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from tools.bounded_identity_run import RequestBudget
from tools.identity_calibration import ExperimentBudget


class BudgetTests(unittest.TestCase):
    def test_parallel_workers_share_one_request_cap(self):
        network = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(b'ok'))
        budget = RequestBudget(2, 60, network)
        def worker(_):
            try:
                with budget(None, 120) as response:
                    return response.read().decode()
            except ExperimentBudget:
                return 'stopped'
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(worker, range(8)))
        self.assertEqual(results.count('ok'), 2)
        self.assertEqual((budget.count, network.call_count), (2, 2))
        self.assertEqual(budget.stop_reason, 'request_budget')
        self.assertTrue(all(call.kwargs['timeout'] <= 60 for call in network.call_args_list))

    def test_time_budget_stops_network_and_does_not_consume_attempt(self):
        network = Mock()
        with patch('tools.bounded_identity_run.time.perf_counter', side_effect=[0, 61]):
            budget = RequestBudget(2, 60, network)
            with self.assertRaises(ExperimentBudget):
                budget(None, 120)
        self.assertEqual((budget.count, network.call_count), (0, 0))
        self.assertEqual(budget.stop_reason, 'time_budget')

    def test_repeated_transport_failure_stops_without_extra_attempts(self):
        network = Mock(side_effect=ConnectionResetError('测试连接中断'))
        budget = RequestBudget(10, 60, network, max_network_failures=2)
        for _ in range(2):
            with self.assertRaises(ConnectionResetError):
                budget(None, 30)
        with self.assertRaises(ExperimentBudget):
            budget(None, 30)
        self.assertEqual((budget.count, network.call_count), (2, 2))
        self.assertEqual(budget.stop_reason, 'network_outage')

    def test_success_resets_consecutive_failures(self):
        network = Mock(side_effect=[ConnectionResetError(), io.BytesIO(b'ok'), ConnectionResetError()])
        budget = RequestBudget(10, 60, network, max_network_failures=2)
        with self.assertRaises(ConnectionResetError):
            budget(None, 30)
        self.assertEqual(budget(None, 30).read(), b'ok')
        with self.assertRaises(ConnectionResetError):
            budget(None, 30)
        self.assertEqual(budget.consecutive_network_failures, 1)
        self.assertIsNone(budget.stop_reason)


if __name__ == '__main__':
    unittest.main()
