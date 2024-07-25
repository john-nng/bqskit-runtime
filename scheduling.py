from task_node import TaskNode
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import colorsys


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

        def default_value():
            return -1
        seen = defaultdict(default_value)

        for t in sorted_tasks:
            task = self.tasks[t.address]
            earliest_start_time = max([self.tasks[parent].created_time + self.tasks[parent].duration for parent in task.parents], default=0)
            task.created_time = earliest_start_time
            task.start_time = task.created_time
            
            base_task_id = get_address_prefix(t.address)
            if seen[base_task_id] != -1:
                # Keep workers that branch off convene at the same worker as it started
                worker_id = seen[base_task_id]
                task.worker = worker_id
                # Assign the task to the worker and update worker end time
                task.start_time = max(task.start_time, worker_end_times[worker_id])
                worker_end_times[worker_id] = task.start_time + task.duration

            else:
                # Find the earliest avaiable worker
                available_worker = min(range(self.num_workers), key=lambda w: worker_end_times[w])
                task.start_time = max(task.start_time, worker_end_times[available_worker])

                # Assign the task to the worker and update worker end time
                worker_end_times[available_worker] = task.start_time + task.duration
                task.worker = available_worker
                seen[base_task_id] = task.worker

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

def PlotSchedule(tasks: Dict[str, TaskNode], num_workers: int, filename: str, regular_time: Optional[int] = None) -> int:
    """ Plot schdule given tasks list"""
    fig, ax = plt.subplots(figsize=(10, num_workers))
    num_blocks = find_num_blocks(tasks=tasks)

    # Coloring Policies dictionary
    color_lookup = populate_color_lookup(tasks=tasks, num_blocks=num_blocks)

    for addr, task in tasks.items():
        start = task.start_time
        duration = task.duration
        if type(task.worker) is str: # task that have not gone through scheduler
            worker = int(task.worker[1:])
        else: # task created from scheduler
            worker = task.worker

        
        color = generate_color(task, color_lookup)
        
        rect = patches.Rectangle((start, worker), duration, 1, edgecolor=color, facecolor=color, label=task.address)
        ax.add_patch(rect)
        

    # Set labels and ticks
    ax.set_xlabel('Time')
    # Center y-ticks
    y_ticks = [i + 0.5 for i in range(num_workers)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'Worker {i}' for i in range(num_workers)])
    
    # Set the x-axis range to cover all tasks
    total_time = max(task.start_time + task.duration for task in tasks.values())
    print(f"{filename} total time: {total_time}")
    # make the x axis as long as a regular run of the workflow
    if regular_time is not None:
        ax.set_xlim(0, regular_time)
    else:
        ax.set_xlim(0, total_time)
    ax.set_ylim(0, num_workers)

    # Add a legend explaining the color coding
    same_color_patch = patches.Patch(color='blue', label='Same colors are correlated subtasks')
    varying_darkness_patch = patches.Patch(color='blue', alpha=0.5, label='Varying darkness differentiates adjacent tasks within a subtask')
    legend = ax.legend(handles=[same_color_patch, varying_darkness_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
    plt.legend = legend

    plt.title(filename)
    plt.tight_layout()
    out_file = f"charts/{filename}.png"
    plt.savefig(out_file)
    plt.close(fig)

    return total_time

def find_num_blocks(tasks) -> int:
    max_block = 0
    for task in tasks.values():
        max_block = max(max_block, get_third_digit(task.address)+1)
    return max_block

def get_address_prefix(address: str) -> str:
    """Get the address prefix up to the third colon."""
    parts = address.split(':')
    return ':'.join(parts[:3])

def get_third_digit(input_string):
    parts = input_string.split(':')
    return int(parts[2])

def populate_color_lookup(
    tasks: Dict[str, TaskNode],
    num_blocks: int
    ) -> dict:
    color_table = {}
    for task in tasks.values():
        key = get_third_digit(task.address)
        base_colors = generate_base_colors(num_blocks)
        base_color = base_colors[get_third_digit(task.address) % num_blocks]
        if not task.parents:
            color_table[get_address_prefix(task.address)] = (0,0,0)
        elif task.parents[0].startswith('-1:0:0'):
            color_table[get_address_prefix(task.address)] = base_color

    return color_table

def generate_color(
    task: TaskNode,
    color_table: dict, 
    ) -> Tuple[float, float, float]:

    if get_address_prefix(task.address) == '-1:0:0':
        return color_table[get_address_prefix(task.address)]

    parent_prefix = get_address_prefix(address=task.parents[0])

    if parent_prefix.startswith('-1:0:0'):
        return color_table[get_address_prefix(task.address)]
    elif len(task.parents) > 1:
        return color_table[get_address_prefix(task.address)]
    else :
        return adjust_brightness(color_table[parent_prefix], brightness_factor=(get_third_digit(task.address)))


### COLOR ASSIGNMENT
def parse_input(input_string):
    parts = input_string.split(':')
    return [int(part) for part in parts]

def generate_base_colors(n):
    base_colors = []
    for i in range(n):
        hue = i / n
        saturation = 1.0  # Lower saturation for softer colors
        lightness = 0.7   # Higher lightness for softer colors
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, lightness)
        base_colors.append((r, g, b))
    return base_colors

def adjust_brightness(color, brightness_factor):
    r, g, b = color
    adjustment = (brightness_factor) * 0.3  # Smaller adjustment
    new_r = max(0, min(1, r * (1 + adjustment)))
    new_g = max(0, min(1, g * (1 + adjustment)))
    new_b = max(0, min(1, b * (1 + adjustment)))
    return new_r, new_g, new_b
