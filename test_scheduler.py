import unittest
from task_node import TaskNode
from scheduling import Scheduler

class TestScheduler(unittest.TestCase):

    def setUp(self):
        # Setup sample tasks for testing
        self.tasks = {
            'A': TaskNode(0, 'worker_1', 'A', 'action_1', duration=3, parents=[]),
            'B': TaskNode(1, 'worker_1', 'B', 'action_2', duration=2, parents=['A']),
            'C': TaskNode(2, 'worker_1', 'C', 'action_3', duration=1, parents=['A']),
            'D': TaskNode(3, 'worker_1', 'D', 'action_4', duration=2, parents=['B', 'C']),
        }
        self.scheduler = Scheduler('test', tasks=self.tasks, num_workers=2)

    def test_topological_sort(self):
        sorted_tasks = self.scheduler.topological_sort()

        # Convert sorted tasks to their addresses for easier verification
        sorted_addresses = [task.address for task in sorted_tasks]

        # Check the order of the sorted tasks
        self.assertIn('A', sorted_addresses)
        self.assertIn('B', sorted_addresses)
        self.assertIn('C', sorted_addresses)
        self.assertIn('D', sorted_addresses)

        # Verify that each task appears after its dependencies
        self.assertLess(sorted_addresses.index('A'), sorted_addresses.index('B'))
        self.assertLess(sorted_addresses.index('A'), sorted_addresses.index('C'))
        self.assertLess(sorted_addresses.index('B'), sorted_addresses.index('D'))
        self.assertLess(sorted_addresses.index('C'), sorted_addresses.index('D'))

if __name__ == '__main__':
    # Run the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScheduler)
    result = unittest.TextTestRunner().run(suite)
    if result.wasSuccessful():
        print("All tests passed successfully!")