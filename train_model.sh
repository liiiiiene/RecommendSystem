#!/bin/bash

# 激活环境
source activate RecSystem || conda activate RecSystem
# 将当前根目录添加到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 进行训练
# echo "正在训练transformer"
# python model/Train_Deep/Train_Deep.py

echo "正在训练fmcross"
python model/Train_Wide/Train_Wide.py

echo "正在训练组合模型"
python model/concat/WideAndDeep.py


