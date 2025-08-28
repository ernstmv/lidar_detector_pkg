#!/bin/bash
set -euo pipefail

# Colors for messages
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

function check_success {
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Error: $1${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ $1 installed successfully.${NC}"
    fi
}

# Check Python version
if ! python3 --version | grep -q "3\.10"; then
  echo -e "${RED}You must use Python == 3.10 for this script.${NC}"
  exit 1
fi
echo -e "${GREEN}✅ Python 3.10 detected.${NC}"

# Install packages with verification
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
check_success "PyTorch + TorchVision + TorchAudio"

pip install -U openmim
check_success "openmim"

pip install mmengine
check_success "mmengine"

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
check_success "mmcv"

# Add PATH if not already present in .zshrc
if ! grep -q 'export PATH=$PATH:$HOME/.local/bin' "$HOME/.zshrc"; then
    echo 'export PATH=$PATH:$HOME/.local/bin' >> "$HOME/.zshrc"
    echo -e "${GREEN}✅ PATH updated in .zshrc${NC}"
else
    echo -e "${GREEN}ℹ️ PATH already configured in .zshrc${NC}"
fi
source "$HOME/.zshrc"

mim install mmdet
check_success "mmdet"

mim install "mmdet3d>=1.1.0"
check_success "mmdet3d"

pip install "numpy<2.0"
check_success "numpy<2.0"

echo -e "${GREEN}🎉 All packages were installed successfully.${NC}"
