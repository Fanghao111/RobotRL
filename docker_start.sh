#!/bin/bash

# ============================================================
# Docker启动脚本 - RL PyBullet项目
# 使用 MIG GPU 设备
# ============================================================

# 配置参数
CONTAINER_NAME="fhz"
IMAGE_NAME="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
MIG_DEVICE="MIG-a2f4d971-d2f8-501f-9bf4-c0f9a3b932e9"
MOUNT_PATH="/data/fanghaozhou"
WORK_DIR="/data/fanghaozhou/RL"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RL PyBullet Docker 启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查是否已存在同名容器
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}发现已存在容器: ${CONTAINER_NAME}${NC}"
    read -p "是否删除并重新创建? (y/n): " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "停止并删除旧容器..."
        docker stop ${CONTAINER_NAME} 2>/dev/null
        docker rm ${CONTAINER_NAME}
    else
        echo -e "${YELLOW}尝试启动已存在的容器...${NC}"
        docker start ${CONTAINER_NAME}
        docker exec -it ${CONTAINER_NAME} bash
        exit 0
    fi
fi

# 拉取镜像（如果不存在）
echo -e "${GREEN}检查Docker镜像...${NC}"
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}$"; then
    echo "拉取镜像: ${IMAGE_NAME}"
    docker pull ${IMAGE_NAME}
fi

# 启动容器
echo -e "${GREEN}启动Docker容器...${NC}"
echo "  - 容器名: ${CONTAINER_NAME}"
echo "  - 镜像: ${IMAGE_NAME}"
echo "  - GPU MIG: ${MIG_DEVICE}"
echo "  - 挂载: ${MOUNT_PATH} -> ${MOUNT_PATH}"

docker run -it \
    --name ${CONTAINER_NAME} \
    --gpus "\"device=${MIG_DEVICE}\"" \
    -v ${MOUNT_PATH}:${MOUNT_PATH} \
    -w ${WORK_DIR} \
    --shm-size=8g \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=${MIG_DEVICE} \
    ${IMAGE_NAME} \
    bash -c "
        echo '========================================' && \
        echo '  初始化RL开发环境...' && \
        echo '========================================' && \
        pip install --upgrade pip && \
        pip install gymnasium stable-baselines3[extra] pybullet numpy opencv-python tensorboard && \
        echo '' && \
        echo '========================================' && \
        echo '  环境准备完成!' && \
        echo '  工作目录: ${WORK_DIR}' && \
        echo '========================================' && \
        exec bash
    "
