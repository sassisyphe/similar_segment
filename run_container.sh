#!/bin/bash

# 显示当前 GPU 状态，只列出显存使用率低于 100MiB 的空闲卡（可根据需要调整阈值）
echo "当前空闲 GPU："
nvidia-smi --query-gpu=index,memory.used,memory.total,name --format=csv | \
  awk -F', ' 'NR>1 && $2+0 < 100 {print "GPU " $1 " : " $4 " (显存 " $3 ")"}'

# 提示用户输入
read -p "请输入要使用的 GPU 编号: " gpu_id

# 运行容器
docker run -it --gpus device="${gpu_id}" --name thr_container \
  -v /disk0/tanghaoran/myproj:/workspace \
  -w /workspace \
  -u $(id -u):$(id -g) \
  -e PS1='\w\$ ' \
  thr_image bash