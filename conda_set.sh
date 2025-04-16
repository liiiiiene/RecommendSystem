#!/bin/bash

# 创建名为RecSystem的conda环境，Python版本为3.12
conda create -n RecSystem python=3.12 -y

# 激活conda环境
source activate RecSystem || conda activate RecSystem

# 安装requirements.txt中的包
pip install -r requirements.txt
