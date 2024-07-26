#!/bin/bash
python prof_tester.py default 2 > logs/compile_out.log 2>&1
python prof_tester.py default 4 >> logs/compile_out.log 2>&1
python prof_tester.py default 8 >> logs/compile_out.log 2>&1
python prof_tester.py default 16 >> logs/compile_out.log 2>&1
python prof_tester.py default 32 >> logs/compile_out.log 2>&1
python prof_tester.py default 64 >> logs/compile_out.log 2>&1
python prof_tester.py default 128 >> logs/compile_out.log 2>&1