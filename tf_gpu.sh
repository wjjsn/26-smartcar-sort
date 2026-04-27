#!/bin/bash
export LD_PRELOAD=/home/wjjsn/code/smartcar/26-smartcar-sort/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12:$LD_PRELOAD
exec /home/wjjsn/code/smartcar/26-smartcar-sort/.venv/bin/python "$@"