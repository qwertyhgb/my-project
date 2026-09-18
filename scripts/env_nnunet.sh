#!/usr/bin/env bash
# 项目内 nnU-Net v2.6.2 运行环境（用法：在项目根目录执行  source scripts/env_nnunet.sh ）
#
# 作用：
#   1) 覆盖 /root/.bashrc 中的全局 nnUNet_* 变量，把 raw/preprocessed/results 全部指向
#      项目内部目录（workdir/ 与 outputs/），与 project_structure.md 及既往项目约定一致；
#   2) 把 PYTHONPATH 钉到 third_party/nnUNet（tag v2.6.2，
#      commit 74ceb6803d10dcee29b2cc481678d3a3d069f281）——lm 环境中可编辑安装的
#      其他 nnunetv2 分支（如 3D-Prostate 项目的 AlignThenRefine fork）会遮蔽固定版本，
#      必须显式前置本项目源码树；
#   3) 校验 Conda 环境与固定源码存在性，避免在错误环境中启动长任务。
#
# 注意：本脚本必须用 source 运行（需要修改当前 shell 的环境变量）。

_ZRF_EXPECTED_PYTHON="/root/anaconda3/envs/lm/bin/python"
_ZRF_ACTUAL_PYTHON="$(command -v python 2>/dev/null || true)"
if [[ "$_ZRF_ACTUAL_PYTHON" != "$_ZRF_EXPECTED_PYTHON" ]]; then
    echo "ERROR: 请先激活 Conda 环境 lm（conda activate lm）。" >&2
    echo "  expected python: $_ZRF_EXPECTED_PYTHON" >&2
    echo "  current python:  ${_ZRF_ACTUAL_PYTHON:-<not found>}" >&2
    return 1 2>/dev/null || exit 1
fi
unset _ZRF_EXPECTED_PYTHON _ZRF_ACTUAL_PYTHON

export ZRF_PROJECT_ROOT="/opt/data/private/lm/my-projects"
export ZRF_DATA_ROOT="/opt/data/private/lm/data/Prostate"
export ZRF_PICAI_ROOT="$ZRF_DATA_ROOT/PI-CAI"
export ZRF_DATASET_ID="605"
export ZRF_DATASET_NAME="Dataset605_PICAI"
export ZRF_NNUNET_SOURCE="$ZRF_PROJECT_ROOT/third_party/nnUNet"

if [[ ! -f "$ZRF_NNUNET_SOURCE/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py" ]]; then
    echo "ERROR: 固定版本 nnU-Net 源码不存在: $ZRF_NNUNET_SOURCE" >&2
    return 1 2>/dev/null || exit 1
fi

# nnUNet 数据目录一律留在项目内：raw/preprocessed → workdir/，results → outputs/
export nnUNet_raw="$ZRF_PROJECT_ROOT/workdir/nnUNet_raw"
export nnUNet_preprocessed="$ZRF_PROJECT_ROOT/workdir/nnUNet_preprocessed"
export nnUNet_results="$ZRF_PROJECT_ROOT/outputs/nnUNet_results"

# 固定 nnunetv2 来源：lm 环境中的可编辑安装（AlignThenRefine fork）不得遮蔽本项目固定源码树
export PYTHONPATH="$ZRF_NNUNET_SOURCE:$ZRF_PROJECT_ROOT/src"
export PYTHONNOUSERSITE=1

mkdir -p "$nnUNet_results" "$nnUNet_preprocessed"

echo "Project-local official nnU-Net 2.6.2 loaded"
echo "  source:       $ZRF_NNUNET_SOURCE"
echo "  dataset:      $ZRF_DATASET_NAME (id=$ZRF_DATASET_ID)"
echo "  raw:          $nnUNet_raw"
echo "  preprocessed: $nnUNet_preprocessed"
echo "  results:      $nnUNet_results"
