import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import os
import copy

class TaskNode:
    """Representation of a Task"""
    def __init__(self, 
        created_time: float, 
        worker: str, 
        address: str, 
        action: str, 
        start_time: float = None, 
        duration: float = None, 
        parents: list[str] = []
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
        self.start_time = start_time
        """Time of task starting. Initially 0 until found Start timestamp in logs."""
        self.started = False
        self.duration = 0
        """Time duration from time of start to time of finish. Initially 0 until found Finish timestamp in logs."""
        self.parents = parents
        """Identifier for direct parent object."""

    def __str__(self):
        # Optional Parent field
        return (f"{self.address} | {self.action} | {self.worker} | Created: {self.created_time} | Started: {self.start_time} | Duration: {self.duration}\nParent: {self.parents}\n")
    
    def __repr__(self):
        return self.__str__()

    
def parse_lines(file_path) -> dict:
    """Parse each line in file into TaskNode Object."""
    with open (file_path, "r") as file:
        lines = file.readlines()

    # Task dictionary that stores all TaskNode Objects
    tasks = {}
    for line in lines:
        line_to_task(line, tasks) # Method will add new objects into tasks automatically

    return tasks

def line_to_task(line: str, tasks: dict) -> None:
    """Logic that converts string into TaskNode object that is stored in tasks."""
    if (not line): return None
    # Seperate line into columns based on "|" dividers
    columns = [col.strip() for col in line.split("|")]
    status = columns[2]
    address = columns[3]
    parents = parse_parents(columns[5]) if columns[5] != "None" else []
    key = ("Task " +address)

    # Creation call is found - create a new task
    if status == "C":
        created_time = float(columns[0])
        worker = columns[1]
        action = columns[4]
        if len(parents) > 1:
            # this is a synchronus type task - when created consider it started too
            start_time = created_time
            tasks[key] = TaskNode(created_time=created_time, start_time=start_time, worker=worker, action=action, address=address, parents=parents)
        else:
            tasks[key] = TaskNode(created_time=created_time, worker=worker, action=action, address=address, parents=parents)

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
    
def parse_parents(address_str):
    """
    Parses the address column from the log line.
    
    Parameters:
    address_str (str): The last column of the log line containing address(es).
    
    Returns:
    list: A list of address strings.
    """
    # Check if the address string starts with '[' and ends with ']'
    if address_str.startswith('[') and address_str.endswith(']'):
        # Remove the brackets and split the string by ', ' to get individual addresses
        addresses = address_str[1:-1].split(', ')
        return [address.strip("'") for address in addresses]
    else:
        # If there's no brackets, return the address string in a list
        return [address_str]

def write_sort_logs(args):
    """"Read file -> Sort on global time -> Make times relative -> Write back file"""
    file_name = args
    out_file_log = f"logs/{file_name[:file_name.find(".")]}_sorted.txt"
    # Read the file content
    with open(f"logs/{file_name}", "r") as file:
        lines = file.readlines()

    sorted_logs = sort_logs(lines)

    # Write back sorted file
    with open(out_file_log, 'w') as file:
        for entry in sorted_logs:
            file.write(' | '.join(entry) + '\n')
    return out_file_log

def repartition_logs(file_name):
    # Read log lines
    with open(file_name, 'r') as file:
        log_lines = file.readlines()

    task_hierarchy = defaultdict(list)
    task_pause_count = defaultdict(int)
    task_paused = defaultdict(bool)
    new_log_lines = []

    for line in log_lines:
        time, worker_id, command, task_id, description, parent_id = map(str.strip, line.strip().split(' | '))
        new_task_id = f"{task_id}:{task_pause_count[task_id]}"
        new_parent_id = f"{parent_id}:{task_pause_count[parent_id]}" if parent_id != "None" else "None"

        if command == 'C':
            task_hierarchy[new_parent_id].append(new_task_id)

        if command == 'P':
            # pause 
            task_paused[new_task_id] = True
            new_pause_line = f"{time} | {worker_id} | {command} | {new_task_id} | {description} | {new_parent_id}"
            new_finish_line = f"{time} | {worker_id} | F | {new_task_id} | {description} | {new_parent_id}"
            new_log_lines.append(new_pause_line.strip())
            new_log_lines.append(new_finish_line.strip())
            
        # Resume
        elif command == 'S' and task_paused[new_task_id]:
            for parent in task_hierarchy[new_task_id]:
                # get first 3 numbers in address - which is the base task id
                base_task_id = ":".join(parent.split(":")[:3])
                # check for the latest sequence for the base task and update the parent address
                updated_parent_task_id = f"{base_task_id}:{task_pause_count[base_task_id]}"
                index = task_hierarchy[new_task_id].index(parent)
                task_hierarchy[new_task_id][index] = updated_parent_task_id

            task_pause_count[task_id] += 1
            new_line = f"{time} | {worker_id} | C | {f"{task_id}:{task_pause_count[task_id]}"} | {description} | {task_hierarchy[new_task_id]}"
            new_log_lines.append(new_line)
            #task_paused[new_task_id] = False

        elif command == 'F' and task_paused[f"{task_id}:{task_pause_count[task_id]-1}"]:
            prev_task_id = f"{task_id}:{task_pause_count[task_id]-1}"
            new_finish_line = f"{time} | {worker_id} | F | {f"{task_id}:{task_pause_count[task_id]}"} | {description} | {task_hierarchy[prev_task_id]}"
            new_log_lines.append(new_finish_line)
            #if task_id in children_tasks_copy:
            #    children_tasks_copy.pop(task_id)
            
        else:
            new_line = f"{time} | {worker_id} | {command} | {new_task_id} | {description} | {new_parent_id}"
            new_log_lines.append(new_line.strip())

    # Attach sub-task IDs to parent tasks
    for i, line in enumerate(new_log_lines):
        parts = line.strip().split(' | ')
        if parts[4].startswith('[') and parts[4].endswith(']'):
            parent_id = parts[4][1:-1]
            new_log_lines[i] = f"{parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} | {parent_id}"

    return new_log_lines

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

def hierarchical_pos(G, root=None, width=1., vert_gap=0.1, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.1, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
        
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)  
    if len(children) != 0:
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    return pos

def plot_graph(file_name:str, tasks:dict[str, TaskNode]) -> None:
    G = nx.DiGraph()

    # Add nodes and edges from TaskNode objects
    for key, task in tasks.items():
        G.add_node(task.address, label=f"{task.address}\n{task.worker}")
        for parent_addr in task.parents:
            G.add_edge(parent_addr, task.address)
        #if not task.parent or task.parent != "None":
        #    G.add_edge(task.parent.address, task.address)

    # Create positions for the nodes based on their start_time and spread out y positions
    y_spacing = 5  # Adjust this value to spread nodes further apart vertically
    pos = {task.address: (task.start_time, i * y_spacing) for i, (key, task) in enumerate(tasks.items())}


    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_size=200, node_color='skyblue', font_size=3, font_color='black',
            font_weight='regular', arrowsize=5)

    plt.xlabel('Start Time')

    # Save the plot to a file
    workflow_name = f"{file_name[file_name.find("logs")+5:file_name.find("_partitioned")]}"
    output_filepath = f"charts/{workflow_name}_dependency_graph.png"
    plt.savefig(output_filepath, format='png')

    plt.close()  # Close the plot to free up resources

def write_repartitioned_log(file_name):
    parsed_log = repartition_logs(file_name)
    workflow_name = f"{file_name[file_name.find("logs")+5:file_name.find("_sorted")]}"
    out_file = f"logs/{workflow_name}_partitioned.txt"
    with open(out_file, 'w') as file:
        for line in parsed_log:
            file.write(line + '\n')
    return out_file

if __name__ == '__main__':
    file_name = sys.argv[1]
    
    sorted_file = write_sort_logs(file_name)
    partitioned_logs = write_repartitioned_log(sorted_file)

    tasks = parse_lines(partitioned_logs)
    print(max([task.duration for task in tasks.values()]))
    
    plot_graph(partitioned_logs, tasks)

