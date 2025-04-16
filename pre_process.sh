#!/bin/bash

# 激活conda环境
source activate RecSystem || conda activate RecSystem

# 将当前根目录添加到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建process_data文件夹
mkdir -p process_data

# 复制gpt_embeddings.csv到process_data文件夹
cp gpt_embeddings.csv process_data/

# 运行数据处理脚本
python load_data/generate_process_data_file.py

echo "预处理完成！" 
