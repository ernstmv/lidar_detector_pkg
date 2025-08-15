#!/bin/bash

if [ "$(python3 --version | grep -c '3\.10')" -lt 1 ]; then
  echo "You must use Python version == 3.10 for this script. If not, your installation will fail or may not work properly."
  exit 1
fi

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install -U openmim

echo "export PATH=$PATH:$HOME/.local/bin" >> .bashrc

pip install mmengine

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

mim install mmdet

mim install "mmdet3d>=1.1.0"

pip install "numpy<2.0"
