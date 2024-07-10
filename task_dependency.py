import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from enum import IntEnum
from bqskit.runtime.address import RuntimeAddress


class TaskType(IntEnum):
    """Type of task."""
    START = 0
    FINISH = 1
    CREATE = 2
    CREATE_SUB = 3

class TaskNode:

    def __init__(self,  worker_id: int, task_id: int, created_time: float, start_time: float = None, duration: float = None, parent: 'TaskNode' = None) -> None:
        #self.parent = parent
        self.worker_id = worker_id
        self.task_id = task_id
        #self.action = action
        self.start_time = 0
        self.created_time = created_time
        self.duration = 0
        self.parent = parent

    def __str__(self):
        parent_info = f"{self.parent.worker_id}, {self.parent.task_id}" if self.parent else "None"
        duration = self.duration if self.duration else "None"
        return (f"Worker {self.worker_id} | Task {self.task_id} | Created Time {self.created_time} | Duration {duration} | Parent {parent_info}")
    
    def __repr__(self):
        return self.__str__()

    
def parse_line(file_path) -> TaskNode:
    # Read lines
    with open (file_path, "r") as file:
        lines = file.readlines()

    tasks = {}

    for line in lines[4:8]:
        data = [s.strip() for s in line.split("|")]
        worker_id = int(data[0].split()[1])
        action = data[1]
        task_id = int(data[2].split()[1])
        task_name = data[3]
        timestamp = data[-1]

        key = (worker_id, task_id)
        
        if action == "Create":
            task_node = TaskNode(worker_id=worker_id, task_id=task_id, created_time=timestamp, parent=None)
            tasks[key] = task_node
        elif action == "Start":
            task_node = tasks.get(key)
            if task_node:
                task_node.start_time = timestamp

        elif action == "Finish":
                # Calculate the duration for the task
                task_node = tasks.get(key)
                if task_node:
                    task_node.duration = task_node.start_time - task_node.created_time

    return tasks

def parse_runtime_obj(input: str) -> RuntimeAddress:
    values = input.split(', ')
    worker_id = int(values[0].split('=')[1])
    mailbox_index = int(values[1].split('=')[1])
    mailbox_slot = int(values[2].split('=')[1])
    
    # Create and return a RuntimeAddress object
    return RuntimeAddress(worker_id, mailbox_index, mailbox_slot)


def grab_sort_logs(args):
    file_name = args
    out_file_log = f"logs/{file_name[:file_name.find(".")]}_sorted.txt"
    # Read the file content
    with open(f"logs/{file_name}", "r") as file:
        lines = file.readlines()

    # Parse the lines to extract columns
    data = []
    for line in lines:
        columns = line.strip().split(' | ')
        data.append(columns)

    last_column = [float(row[-1]) for row in data]
    min_value = min(last_column)

    for row in data:
        row[-1] = str(float(row[-1]) - min_value)

    # Sort the data by completion time (last column)
    sorted_data = sorted(data, key=lambda x: float(x[-1]))

    map(lambda x: x - min(data[-1]), data[-1])

    with open(out_file_log, 'w') as file:
        for entry in sorted_data:
            file.write(' | '.join(entry) + '\n')
    return out_file_log

if __name__ == '__main__':
    file_name = sys.argv[1]
    # Log the sorted original log file
    
    print(parse_line(grab_sort_logs(sys.argv[1])))
