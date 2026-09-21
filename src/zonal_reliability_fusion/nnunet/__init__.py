"""nnU-Net-first 集成层。

本包是项目对固定版本 nnU-Net（third_party/nnUNet，v2.6.2）的**唯一**扩展点，
只保留三样研究必需的东西：

1. PI-CAI Focal + CE 损失（``trainers.PiCAIFocalCrossEntropyLoss``）；
2. Gaussian blur 的 NoFFT 兼容修复（``trainers`` 中的 augmentation mixin）；
3. 一个基于原生 nnU-Net 网络的轻量 spatial modality reliability gate
   （``networks``），以及 anatomy variant 所需的 PZ/TZ 通道增强边界
   （``transforms``）。

planning/preprocessing、dataloader、patch sampling、augmentation 主体、deep
supervision、optimizer/scheduler、training loop、checkpoint/resume、validation、
sliding-window inference 与 prediction export 全部由 nnU-Net 负责，不在此重复实现。
"""

from __future__ import annotations

from pathlib import Path

#: 项目根目录（.../my-projects）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
#: 固定 nnU-Net 源码根（只读）
FIXED_NNUNET_ROOT = PROJECT_ROOT / "third_party" / "nnUNet"
#: nnU-Net 2.6.2 对 dynamic-network-architectures 的官方约束
DNA_REQUIRED_RANGE = ">=0.4.1,<0.5"
#: 本项目 plans 实际使用的网络类（Dataset605/606 的 3d_fullres 均为该原生类）
PLAIN_CONV_UNET_DOTTED_PATH = (
    "dynamic_network_architectures.architectures.unet.PlainConvUNet"
)


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in text.split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_fixed_nnunet_runtime(*, strict: bool = True, quiet: bool = False) -> dict:
    """明确校验当前进程实际使用的 nnU-Net / dynamic-network-architectures 运行时。

    **只检查、不升级、不覆盖任何包**。返回一个 info dict，并在必要时打印如实说明：

    - ``nnunetv2`` 必须解析到项目固定的 ``third_party/nnUNet``（v2.6.2）。若解析到别处
      （例如 lm 环境里来自 PENGWIN 的可编辑 nnUNetv2 2.5），说明没有 ``source scripts/env_nnunet.sh``
      或固定源码被遮蔽；``strict=True`` 时直接报错，避免用错误分支训练/预测。
    - 实际使用的 ``PlainConvUNet`` 必须能由 ``pydoc`` **直接定位**（这样
      ``get_network_from_plans`` 不会退化为在 DNA 包内递归扫描类），并报告其文件路径。
    - 如实报告 ``dynamic-network-architectures`` 版本是否满足 2.6.2 的 ``>=0.4.1,<0.5``。
      **不满足时只 WARNING**（当前仅 PlainConvUNet 路径经合成测试验证可用），绝不宣称依赖完全满足。
    """
    import inspect
    import pydoc

    import nnunetv2

    info: dict[str, object] = {}
    nnunetv2_file = str(Path(nnunetv2.__file__).resolve())
    info["nnunetv2_file"] = nnunetv2_file
    info["nnunetv2_is_fixed"] = nnunetv2_file.startswith(str(FIXED_NNUNET_ROOT))
    if strict and not info["nnunetv2_is_fixed"]:
        raise RuntimeError(
            "nnunetv2 未解析到项目固定源码 third_party/nnUNet（v2.6.2）：\n"
            f"  resolved: {nnunetv2_file}\n"
            f"  expected under: {FIXED_NNUNET_ROOT}\n"
            "请先执行：cd 项目根目录 && conda activate lm && source scripts/env_nnunet.sh；"
            "否则可能加载到环境中其他 nnunetv2 分支（如可编辑安装的 2.5 fork）。"
        )

    # PlainConvUNet 明确路径检查：直接定位成功 => 不触发递归扫描回退
    arch_cls = pydoc.locate(PLAIN_CONV_UNET_DOTTED_PATH)
    if arch_cls is None:
        raise RuntimeError(
            f"无法直接定位 {PLAIN_CONV_UNET_DOTTED_PATH}；"
            "get_network_from_plans 可能退化为递归扫描，且当前 DNA 安装不可用。"
        )
    info["plain_conv_unet_file"] = inspect.getfile(arch_cls)

    import importlib.metadata as importlib_metadata

    try:
        dna_version = importlib_metadata.version("dynamic-network-architectures")
    except Exception:  # noqa: BLE001 - 元数据缺失时如实标注 unknown
        dna_version = "unknown"
    info["dna_version"] = dna_version
    satisfies = dna_version != "unknown" and (0, 4, 1) <= _parse_version(
        dna_version
    ) < (0, 5, 0)
    info["dna_satisfies_nnunet_2_6_2"] = satisfies
    info["dna_required_range"] = DNA_REQUIRED_RANGE

    if not quiet:
        if not satisfies:
            print(
                "[nnunet-runtime][WARNING] dynamic-network-architectures "
                f"{dna_version} 不满足 nnU-Net 2.6.2 的官方约束 {DNA_REQUIRED_RANGE}。\n"
                "  说明：lm 环境中 nnUNetv2 2.5（来自 PENGWIN 的可编辑安装）要求 DNA<0.4，"
                "与 2.6.2 的 >=0.4.1 冲突；本项目**不升级/不覆盖** DNA。\n"
                "  当前边界：仅 PlainConvUNet 路径（Dataset605/606 的 3d_fullres）经合成测试验证可用；"
                "不能据此宣称依赖完全满足。改用 ResidualEncoderUNet 等 DNA>=0.4 架构前必须先解决该冲突。\n"
                f"  nnunetv2: {nnunetv2_file}\n"
                f"  PlainConvUNet: {info['plain_conv_unet_file']}"
            )
        else:
            print(
                f"[nnunet-runtime] OK: nnunetv2={nnunetv2_file}; "
                f"DNA={dna_version} 满足 {DNA_REQUIRED_RANGE}; "
                f"PlainConvUNet={info['plain_conv_unet_file']}"
            )
    return info
