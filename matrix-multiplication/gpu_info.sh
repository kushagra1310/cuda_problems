#!/bin/bash
# gpu_info.sh — print a single-line, comma-safe identifier for the current
# GPU so it can be stamped into every row of a benchmark/profiling CSV.
#
# Usage:
#   GPU_TAG=$(./gpu_info.sh)
#
# Output format: "<name>|sm_<major><minor>"
#   e.g. "NVIDIA A100-SXM4-80GB|sm_80"
#
# Run this once per machine/architecture you benchmark on, and keep the
# resulting CSVs — the analysis script groups on this column to produce
# the "multi-architecture scaling" plots the assignment asks for.

set -e

NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/,//g')
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
SM="sm_$(echo "$CC" | tr -d '.')"

echo "${NAME}|${SM}"