import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class TaskNode:

    def __init__(self,  worker_id: int ) -> None:
        #self.parent = parent
        self.worker_id = worker_id



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

    # Sort the data by completion time (last column)
    sorted_data = sorted(data, key=lambda x: float(x[-1]))

    with open(out_file_log, 'w') as file:
        for entry in sorted_data:
            file.write(' | '.join(entry) + '\n')

if __name__ == '__main__':
    file_name = sys.argv[1]
    # Log the sorted original log file
    grab_sort_logs(sys.argv[1])
