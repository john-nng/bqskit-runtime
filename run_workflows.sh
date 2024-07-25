#!/bin/bash
python prof_tester.py qsearch 3 2 > logs/workflows_output.log 2>&1
python prof_tester.py qsearch 3 4 >> logs/workflows_output.log 2>&1
python prof_tester.py qsearch 3 8 >> logs/workflows_output.log 2>&1
python prof_tester.py qsearch 3 16 >> logs/workflows_output.log 2>&1
python prof_tester.py qsearch 3 32 >> logs/workflows_output.log 2>&1

python task_dependency.py qsearch_3_2.txt >> logs/workflows_output.log 2>&1
python task_dependency.py qsearch_3_4.txt >> logs/workflows_output.log 2>&1
python task_dependency.py qsearch_3_8.txt >> logs/workflows_output.log 2>&1
python task_dependency.py qsearch_3_16.txt >> logs/workflows_output.log 2>&1
python task_dependency.py qsearch_3_32.txt >> logs/workflows_output.log 2>&1