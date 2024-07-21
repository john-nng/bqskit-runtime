from task_node import TaskNode
from typing import List, Dict
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


class Scheduler():
    """ Scheduler Class that handles task scheduling given task dependency graphs"""
    def __init__(self, name: str, tasks: Dict[str, TaskNode], num_workers: int) -> None:
        self.tasks = tasks
        self.num_workers = num_workers
        self.name = name

    def Schedule(self) -> Dict[str, TaskNode]:
        """Schdule will assign task in topological order to the earliest avaiable worker (ASAP algorithim)
            Returns dictionary of tasks with optimized start times"""
        sorted_tasks = self.TopologicalSort()
        worker_end_times = [0] * self.num_workers # Track when each worker is free
        #TODO latency not working as intended
        comm_latency = 0.01 # account for the time it takes to send a recive messages for a task to start after its been created
        #TODO make base task start and end in same workers
        for t in sorted_tasks:
            task = self.tasks[t.address]
            earliest_start_time = max([self.tasks[parent].created_time + self.tasks[parent].duration for parent in task.parents], default=0)
            task.created_time = earliest_start_time
            task.start_time = task.created_time + comm_latency
            
            # Find the earliest avaiable worker
            available_worker = min(range(self.num_workers), key=lambda w: worker_end_times[w])
            task.start_time = max(task.start_time, worker_end_times[available_worker])

            # Assign the task to the worker and update worker end time
            worker_end_times[available_worker] = task.start_time + task.duration
            task.worker = available_worker

        return self.tasks
        

    def TopologicalSort(self) -> List[TaskNode]:
        """Given a task dependency graph, topologically"""
        if not self.tasks:
            return []

        degree = {task: 0 for task in self.tasks}
        for task in self.tasks.values():
            for parent in task.parents:
                if parent != "None":
                    degree[task.address] += 1

        # Add to queue if task has no parents - no dependencies
        queue = deque([task for task in self.tasks.values() if degree[task.address] == 0])
        sorted_tasks = []

        while queue:
            current_task = queue.popleft()
            sorted_tasks.append(current_task)

            for dependent_task in self.tasks.values():
                # Check if current task is a parent of any other task
                if current_task.address in dependent_task.parents:
                    degree[dependent_task.address] -= 1
                    if degree[dependent_task.address] == 0:
                        queue.append(dependent_task)

        return sorted_tasks

def PlotSchedule(tasks: Dict[str, TaskNode], num_workers: int, filename: str) -> None:
    """ Plot schdule given tasks list"""
    fig, ax = plt.subplots(figsize=(10, num_workers))

    # Dictionary to keep track of parent colors
    task_colors = {}

    for addr, task in tasks.items():
        start = task.start_time
        duration = task.duration
        if type(task.worker) is str:
            worker = int(task.worker[1:])
            if worker == -1:
                worker = 1
        else:
            worker = task.worker

        # Each base task will have a unique color
        address_prefix = get_address_prefix(task.address)
        if address_prefix not in task_colors:
            task_colors[address_prefix] = get_random_color()
        color = task_colors[address_prefix]
        
        rect = patches.Rectangle((start, worker), duration, 1, edgecolor=color, facecolor=color, label=task.address)
        ax.add_patch(rect)
        
        # Annotate the task with its address
        #ax.text(start + duration / 2, worker + 0.5, task.address, ha='center', va='center', color='black')

    # Set labels and ticks
    ax.set_xlabel('Time')
    ax.set_ylabel('Workers')
    # Center y-ticks
    y_ticks = [i + 0.5 for i in range(num_workers)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'Worker {i}' for i in range(num_workers)])
    
    # Set the x-axis range to cover all tasks
    max_time = max(task.start_time + task.duration for task in tasks.values())
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, num_workers)

    plt.title(filename)
    out_file = f"charts/{filename}.png"
    plt.savefig(out_file)
    plt.close(fig)


### COLOR ASSIGNMENT
def get_address_prefix(address: str) -> str:
    """Get the address prefix up to the third colon."""
    parts = address.split(':')
    return ':'.join(parts[:3])

def get_random_color():
    """Generate a random RGB color."""
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)

def generate_color(base_color, variation):
    """Generate a color variant based on a base color and variation index"""
    h, l, s = base_color
    l = min(1.0, l + variation * 0.1)
    return (h, l, s)

def address_to_base_color(address):
    """Generate a base color based on the first three digits of the address"""
    parts = address.split(':')
    if len(parts) < 3:
        return (0, 0.5, 0.5)  # Default color if address format is unexpected
    key = ':'.join(parts[:3])
    hash_value = hash(key)
    hue = (hash_value % 360) / 360.0
    return (hue, 0.5, 0.5)

def get_color(address, parent_colors):
    """Get color for a task, variant of parent's color if applicable"""
    base_color = address_to_base_color(address)
    if parent_colors:
        # Generate a variant color based on the parent's color
        return generate_color(parent_colors[0], len(parent_colors))
    return base_color
