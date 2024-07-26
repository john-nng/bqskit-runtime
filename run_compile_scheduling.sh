#!/bin/bash
python task_dependency.py default_2.txt >> logs/compile_out.log 2>&1
python task_dependency.py default_4.txt >> logs/compile_out.log 2>&1
python task_dependency.py default_8.txt >> logs/compile_out.log 2>&1
python task_dependency.py default_16.txt >> logs/compile_out.log 2>&1
python task_dependency.py default_32.txt >> logs/compile_out.log 2>&1
python task_dependency.py default_64.txt >> logs/compile_out.log 2>&1
python task_dependency.py default_128.txt >> logs/compile_out.log 2>&1