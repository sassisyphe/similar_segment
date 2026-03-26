#!/bin/bash
# run.sh

# 进入工作目录
cd /workspace

# 将脚本的所有输出（标准输出和标准错误）输出到 run_full.log
exec > run_full.log 2>&1

# 激活虚拟环境
source /workspace/venv/bin/activate

# 强制 Python 使用无缓冲输出，确保日志实时更新
PYTHON="python -u"

# 允许通过参数指定运行的模块，例如：bash run.sh 4 （只运行模块4）
# 如果不传参数，默认运行 1 2 3 4
if [ $# -eq 0 ]; then
    MODULES="1 2 3 4"
else
    MODULES="$@"
fi

echo "=================================================="
echo "Starting Execution at $(date)"
echo "Modules to run: $MODULES"
echo "=================================================="

for mod in $MODULES; do
    case $mod in
        1)
            echo '--- Running Module 1: Download Data ---'
            $PYTHON module1_download.py
            ;;
        2)
            echo '--- Running Module 2: Generate Segments ---'
            $PYTHON module2_sample.py
            ;;
        3)
            echo '--- Running Module 3: Similarity Search ---'
            $PYTHON module3_search.py
            ;;
        4)
            echo '--- Running Module 4: Analysis & Visualization ---'
            $PYTHON module4_analyze.py
            ;;
        *)
            echo "Unknown module: $mod"
            ;;
    esac
done

echo "=================================================="
echo "Execution finished at $(date)"
echo "=================================================="
