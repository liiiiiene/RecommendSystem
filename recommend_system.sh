#!/bin/bash

# 激活环境
source activate RecSystem || conda activate RecSystem
# 将当前根目录添加到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 生成推荐列表
echo "生成推荐列表"
python model/System/recommend_dict.py
