#!/bin/bash

cd .venv/bin/python3 main.py > .research.log 2>&1 &
echo $! > .pid.research
