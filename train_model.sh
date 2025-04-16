#!/bin/bash

# 激活环境
source activate RecSystem || conda activate RecSystem
# 将当前根目录添加到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 进行训练
echo "正在训练组合模型"
python model/concat/WideAndDeep.py 
