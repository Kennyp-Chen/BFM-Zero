#!/bin/bash

# TensorBoard 启动脚本
# 用法: ./start_tensorboard.sh [log_dir]

# 默认日志目录
DEFAULT_LOG_DIR="results/23dof-bfmzero-isaac/tensorboard_logs"

# 使用传入的参数或默认值
LOG_DIR=${1:-$DEFAULT_LOG_DIR}

echo "启动 TensorBoard..."
echo "日志目录: $LOG_DIR"
echo "访问地址: http://localhost:6006"
echo "按 Ctrl+C 停止"

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "警告: 日志目录 $LOG_DIR 不存在"
    echo "请先运行训练脚本生成日志"
    echo "或者指定正确的日志目录: ./start_tensorboard.sh <path_to_logs>"
    exit 1
fi

# 启动 TensorBoard
tensorboard --logdir="$LOG_DIR" --port=6006 --host=0.0.0.0
