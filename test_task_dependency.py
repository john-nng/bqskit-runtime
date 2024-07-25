import unittest
from task_node import TaskNode
import task_dependency as td

class TestLinetoTask(unittest.TestCase):

    def setUp(self):
        # Setup sample tasks for testing
        self.tasks = {
            "2:2:35:1": TaskNode(17.31446442590095, 'W10', '2:2:35:1', '_sub_do_work', 17.31446442590095, 17.31446442590095-17.31516917794943, ['10:1:0:0'])
            
        }

    def test_line_to_task(self):
        test_tasks = {}
        inputs = [
            "17.31446442590095 | W10 | C | 2:2:35:1 | _sub_do_work | ['10:1:0:0']",
            "17.31516917794943 | W10 | F | 2:2:35:1 | _sub_do_work | ['10:1:0:0']"
        ]
        for input in inputs:
            td.line_to_task(line=input, tasks=test_tasks)
        
        self.assertEqual(str(test_tasks['2:2:35:1']), str(self.tasks['2:2:35:1']))


if __name__ == '__main__':
    # Run the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinetoTask)
    result = unittest.TextTestRunner().run(suite)
    if result.wasSuccessful():
        print("All tests passed successfully!")