[working-directory: '.']
train-model:
    LD_PRELOAD=.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12  uv run main.py --framework tensorflow