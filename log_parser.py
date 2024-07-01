import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class LogData:

    def __init__(self, run_time: float, idle_time: float, num_tasks: float, timeline: list[tuple[str, float]]):
        self.run_time = run_time
        self.idle_time = idle_time
        self.num_tasks = num_tasks
        self.timeline = timeline

    def __str__(self) -> str:
        return f"Run Time: {self.run_time} Idle Time: {self.idle_time} Num Tasks: {self.num_tasks}"
    
    def plot_timeline(self, axes: plt.Axes, y_height:float, width: float):
        color_map = {
            "idle": "r",
            "instantiate": "g",
            "qsd": "c",
            "decompose": "m",
        }
        x = 20
        for item in self.timeline:
            task_name, start, duration = item
            color = color_map.get(task_name, "b")
            rect = patches.Rectangle((start, y_height), width=duration, height=width, edgecolor=color, facecolor=color)
            axes.add_patch(rect) 
            x = start + duration
            
        return x



class Parser:

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        # self.sent_tasks = 0
        # self.completed_tasks = 0
        self.prog_start_time = None
        # self.run_time = 0
        # self.idle_time = 0
        self.start_task_time = None
        self.timeline = []
    
    def get_log_data(self):
        return LogData(0, 0, 0, self.timeline)
        # return LogData(self.run_time, self.idle_time, max(self.sent_tasks, self.completed_tasks), self.timeline)
    
    def split_log_line(line: str):
        if not line.startswith("Worker"):
            return []
        log_lines = line.split("Worker ")
        all_arrs = []
        for log_line in log_lines:
            if len(log_line) == 0:
                continue
            arr = log_line.split("|")
            arr = [x.strip() for x in arr]
            arr[0] = int(arr[0])
            arr[-1] = float(arr[-1])
            all_arrs.append(arr)
        return all_arrs

    def parse_line(self, worker_id: int, task_type: str, task_name: str, time: float):
        if worker_id != self.worker_id:
            return
        
        if self.prog_start_time is None:
            self.prog_start_time = time
            assert(task_type.startswith("start"))

        if task_type == "finish step": # Actually performing a step
            assert task_name == self.cur_task
            # Add to timeline (task_name, start_time, duration_time)
            time_obj = (task_name, self.start_task_time, time - self.start_task_time - self.prog_start_time)
            # print(time_obj)
            self.timeline.append(time_obj)
        elif task_type == "start step": # Starting a step
            # Track start of task time
            self.start_task_time = time - self.prog_start_time
            self.cur_task = task_name
        elif task_type == "start idle":
            # Track start of idle
            self.start_task_time = time - self.prog_start_time
            self.cur_task = "idle"
        elif task_type == "stop idle" or task_type == "finish idle":
            assert self.cur_task == "idle"
            # Add to timeline ("idle", start_time, duration_time)
            time_obj = (task_name, self.start_task_time, time - self.start_task_time - self.prog_start_time)
            # print(time_obj)
            self.timeline.append(time_obj)
        

def parse_worker(args) -> LogData:
    worker_id, file_name = args
    parser = Parser(worker_id)
    with open(f"logs/{file_name}", "r") as f_obj:
        for line in f_obj.readlines():
            lines = Parser.split_log_line(line)
            for line in lines:
                parser.parse_line(*line)
    return parser.get_log_data()


if __name__ == '__main__':
    file_name = sys.argv[1]
    num_workers = int(file_name.split('_')[-1].split('.')[0])
    out_file_name = f"charts/{file_name}.png"
    fig, axes = plt.subplots(1,1, figsize=(10, 10))

    with mp.Pool(processes=num_workers) as pool:
        log_data = pool.map(parse_worker, [(i, file_name) for i in range(num_workers)])

    max_x = 20
    for j, data in enumerate(log_data):
        x = data.plot_timeline(axes, j, 1)
        if x > max_x:
            max_x = x

    axes.set_ybound(0, num_workers)
    axes.set_xbound(0, x)

    # Add titles
    task = file_name.split('_')[0]
    axes.set_title(f"Worker Task Timeline: {task} {num_workers} Worker(s)", fontsize=20)
    axes.set_xlabel('Time (seconds)', fontsize=14)
    axes.set_ylabel('Worker ID', fontsize=14)
    # axes.set_yticks() Add ticks later

    #Add Color Legend
    color_map = {
            "idle": "r",
            "instantiate": "g",
            "qsd": "c",
            "decompose": "m",
            "send_message": "b",
        }
    legend_field = [patches.Patch(color=color, label=label) for label, color in color_map.items()]
    axes.legend(handles = legend_field, bbox_to_anchor=(1, 1.35), loc='upper right', fontsize=14)
    fig.subplots_adjust(top=0.75)  # Adjust right margin to make space for the legend

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)

    fig.savefig(out_file_name)