import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class TaskNode:
    """Representation of a Task"""
    def __init__(self, 
        created_time: float, 
        worker: str, 
        address: str, 
        action: str, 
        start_time: float = None, 
        duration: float = None, 
        parent: 'TaskNode' = None
    ) -> None:
        """Create the TaskNode Object with task address, worker id, parent address, along with a created timestamp."""
        self.worker = worker
        """Worker who is handling this task."""
        self.created_time = created_time
        """Time of task creation."""
        self.address = address
        """Unique identifier of Task in the form of a RuntimeAddress"""
        self.action = action
        """Type of Task ex: instansiate, sub_do_work"""
        self.start_time = 0
        """Time of task starting. Initially 0 until found Start timestamp in logs."""
        self.started = False
        self.duration = 0
        """Time duration from time of start to time of finish. Initially 0 until found Finish timestamp in logs."""
        self.parent = parent
        """Identifier for direct parent object."""

    def __str__(self):
        # Optional Parent field
        parent_info = f"{self.parent}" if self.parent else "None"
        return (f"Task {self.address} | {self.action} | {self.worker} | Parent {parent_info}\nCreated: {self.created_time} | Started: {self.start_time} | Duration: {self.duration} \n")
    
    def __repr__(self):
        return self.__str__()

    
def parse_lines(file_path) -> dict:
    """Parse each line in file into TaskNode Object."""
    with open (file_path, "r") as file:
        lines = file.readlines()

    # Task dictionary that stores all TaskNode Objects
    tasks = {}
    for line in lines:
        parse_line(line, tasks) # Method will add new objects into tasks automatically

    return tasks

def parse_line(line: str, tasks: dict) -> None:
    """Logic that converts string into TaskNode object that is stored in tasks."""
    if (not line): return None
    # Seperate line into columns based on "|" dividers
    columns = [col.strip() for col in line.split("|")]
    status = columns[2]
    address = columns[3]
    parent = columns[5]
    key = (address, parent)

    # Creation call is found - create a new task
    if status == "C":
        created_time = float(columns[0])
        worker = columns[1]
        action = columns[4]
        tasks[key] = TaskNode(created_time=created_time, worker=worker, action=action, address=address, parent=parent)
    # Start call is found - update object's start_time field
    elif status == "S":
        # Existing Task - grab from tasks dict
        # May see S again before a F, if already started then dont set start time again
        start_time = float(columns[0])       
        if not tasks[key].started:
            tasks[key].start_time = start_time
            tasks[key].started = True
    # Finish call is found - update object's duration field
    elif status == "F":
        # Existing Task - grab from tasks dict
        finish_time = float(columns[0])
        tasks[key].duration = finish_time - tasks[key].start_time 
    
    else:
        ValueError("Could not read line properly")
        return None

def process_logs(args):
    """"Read file -> Sort on global time -> Make times relative -> Write back file"""
    file_name = args
    out_file_log = f"logs/{file_name[:file_name.find(".")]}_processed.txt"
    # Read the file content
    with open(f"logs/{file_name}", "r") as file:
        lines = file.readlines()

    sorted_logs = sort_logs(lines)

    # Write back processed file
    with open(out_file_log, 'w') as file:
        for entry in sorted_logs:
            file.write(' | '.join(entry) + '\n')
    return out_file_log

def sort_logs(logs: list[str])-> list[str]:
    # Parse the lines to extract columns
    data = []
    for line in logs:
        columns = line.strip().split(' | ')
        data.append(columns)

    time_column = [float(row[0]) for row in data]
    min_value = min(time_column)

    for row in data:
        row[0] = str(float(row[0]) - min_value)

    # Sort the data by completion time (first column)
    sorted_data = sorted(data, key=lambda x: float(x[0]))

    return sorted_data

if __name__ == '__main__':
    file_name = sys.argv[1]
    
    processed_file = process_logs(file_name)
    tasks = parse_lines(processed_file)
    print(tasks)
