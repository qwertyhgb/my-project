"""Stage 1 解剖先验的**输入/输出布局**与来源标记。

这个模块回答一个具体问题：**「predicted anatomy prior」到底存放在哪里、以什么形式、
由谁生成？** 因为新主线要求 lesion model 只吃**预测**先验，任何把预测先验与 GT 标签混淆的
路径都会让整个研究结论失效，所以这一层必须显式、可审计。

三层数据的区分（禁止混淆）
--------------------------
======================  ==========================  ================================
层                      内容                        在本项目中的角色
======================  ==========================  ================================
raw inputs              T2W / ADC / HBV             模型输入（只读原始医学数据）
training labels         lesion GT（Dataset605/606） 训练监督与验证参照
predicted priors        P(WG) / P(PZ) / P(TZ)       Stage-1 模型对该病例自身 MRI 的预测
oracle labels           GT WG/PZ/TZ                 仅上界分析，必须标记 ``ORACLE_GT``
======================  ==========================  ================================

为什么必须是 soft probability 而不是硬 mask
------------------------------------------
1. 硬 mask 的边界错误会**直接删除**跨越 WG 边界的病灶（见 :mod:`..lesion.roi` 的设计约束）；
2. 下游的 ROI 与 zone 条件化都需要可微的软证据，而不是 0/1 台阶；
3. 概率图保留了「anatomy 不确定」这一信息，uncertain 区域可以走 residual path 而不是被裁掉。

导出契约
--------
Stage-1 的训练与验证使用 nnU-Net 原生 ``--export-validation-probabilities``，它写出：

- ``<case>.npz``——唯一键 ``probabilities``，形状 ``(3, Z, Y, X)``，顺序与
  ``regions_class_order`` 一致（WG, PZ, TZ）；
- ``<case>.pkl``——nnU-Net 原生物理元数据（``sitk_stuff`` / ``spacing`` /
  ``shape_before_cropping``）；
- ``<case>.nii.gz``——原生 region 导出（按顺序阈值写入，后写区域覆盖前写区域）。

**硬导出只供评估与展示**：任何下游病灶模型读的是 ``.npz`` 里的 soft probability，
不是 ``.nii.gz`` 里的硬标签。
"""

from __future__ import annotations

from pathlib import Path

from zonal_reliability_fusion.anatomy.contracts import (
    ANATOMY_CONFIGURATION,
    ANATOMY_DATASET,
    ANATOMY_PREDICTION_SUFFIXES,
)

#: 三个 region 头在 ``probabilities`` 数组第 0 维上的顺序
ANATOMY_PROBABILITY_CHANNELS = ("WG", "PZ", "TZ")

#: anatomy 先验的默认存放根目录（相对项目根）。目录由用户创建，本模块不自动创建。
DEFAULT_ANATOMY_PRIOR_DIR = "workdir/anatomy_priors"

#: 先验来源标记——与 ``evaluation.anatomy_metrics`` 共用同一套取值，避免两处定义漂移
PRIOR_LAYOUT = {
    "produced_by": (
        f"{ANATOMY_DATASET} / {ANATOMY_CONFIGURATION} 的冻结 anatomy 模型，"
        "对该病例**自身** MRI 的原生滑窗推理输出"
    ),
    "required_suffixes": ANATOMY_PREDICTION_SUFFIXES,
    "probability_array": (
        "npz 唯一键 'probabilities'，形状 (3, Z, Y, X)，通道顺序 WG/PZ/TZ，float，值域 [0,1]"
    ),
    "soft_not_binary": (
        "下游只读 soft probability；ordered-region 硬导出仅供评估与展示；不能恢复重叠 WG head"
    ),
    "forbidden": (
        "禁止使用 GT WG/PZ/TZ 作为 lesion 模型的推理输入；"
        "禁止手工编辑某个病例的先验；禁止在看到 test 结果后重新生成先验"
    ),
}


def prior_directory(root: str | Path, *, dataset_name: str = "Dataset605_PICAI") -> Path:
    """返回某数据集解剖先验的目录（**只计算路径，不创建目录**）。

    目录布局：``<root>/<dataset_name>/prototype_prior/``。是否存在由用户决定；调用方负责
    在缺失时给出清晰的错误信息，而不是静默生成空先验。
    """
    return Path(root) / str(dataset_name) / "prototype_prior"


def case_prior_paths(directory: str | Path, case_id: str) -> dict[str, Path]:
    """返回一个病例的全部先验文件路径（不检查存在性）。"""
    folder = Path(directory)
    return {
        "probabilities_npz": folder / f"{case_id}.npz",
        "properties_pkl": folder / f"{case_id}.pkl",
        "native_export": folder / f"{case_id}.nii.gz",
    }


PREDICTED_DATASET = "Dataset608_PICAI_PredictedAnatomy"
PRIOR_MODES = ("IN_SAMPLE_PRED", "OOF_PRED", "HELD_OUT_PRED", "EXTERNAL_PRED")


def validate_predicted_prior_dataset(document, dataset_name):
    """Fail closed on old zonal inputs, unknown provenance and patient leakage.

    This validates metadata; materialization additionally validates every array.
    Anatomy-training patient IDs are required per checkpoint, including OOF models.
    """
    if dataset_name != PREDICTED_DATASET:
        raise ValueError("anatomy fusion requires independent Dataset608; never Dataset606")
    channels = document.get("channel_names", {})
    if list(channels.values()) != ["T2W", "ADC", "HBV", "noNorm", "noNorm", "noNorm"]:
        raise ValueError("expected T2W/ADC/HBV/WG/PZ/TZ channels; priors use noNorm")
    if document.get("labels") != {"background": 0, "lesion": 1}:
        raise ValueError("expected binary lesion labels")
    contract = document.get("predicted_anatomy_contract", {})
    if contract.get("channel_order") != ["WG", "PZ", "TZ"]:
        raise ValueError("missing predicted anatomy channel order")
    if contract.get("source") != "predicted_prior":
        raise ValueError("only predicted priors are permitted")
    train, val = contract.get("train_cases", []), contract.get("val_cases", [])
    patient = lambda cid: str(cid).split("_")[0]
    if not train or not val or set(map(patient, train)) & set(map(patient, val)):
        raise ValueError("missing split or patient leakage")
    records = contract.get("cases", {})
    expected = set(train) | set(val)
    if (set(train) & set(val) or len(expected) != len(train) + len(val)
            or set(records) != expected or document.get("numTraining") != len(expected)):
        raise ValueError("prior provenance must cover the exact complete case set")
    for cid, record in records.items():
        mode = record.get("prior_mode")
        if mode not in PRIOR_MODES or record.get("case_id") != cid:
            raise ValueError(f"{cid}: invalid prior mode or case identity")
        for key in ("checkpoint", "trainer", "source_split", "geometry"):
            if not record.get(key):
                raise ValueError(f"{cid}: missing provenance {key}")
        trained = record.get("anatomy_training_patient_ids")
        if not isinstance(trained, list) or not trained:
            raise ValueError(f"{cid}: missing anatomy training scope")
        was_seen = patient(cid) in set(map(str, trained))
        if was_seen != (mode == "IN_SAMPLE_PRED"):
            raise ValueError(f"{cid}: prior mode contradicts anatomy training scope")
        if cid in val and mode != "HELD_OUT_PRED":
            raise ValueError(f"{cid}: validation requires held-out prediction")
        if cid in train and mode not in ("IN_SAMPLE_PRED", "OOF_PRED", "HELD_OUT_PRED"):
            raise ValueError(f"{cid}: invalid training prior mode")
    return contract


def materialize_predicted_dataset(source_raw, prior_dir, model_folder, output_raw, split_file,
                                  *, no_progress=False):
    """User-run Dataset608 materializer. No resampling; native preprocessing follows.

    Single frozen Stage-1 model: explicit in-sample train and held-out validation.
    OOF metadata is supported by the validator, not claimed by this materializer.
    Refuses existing output, checks each case geometry before writing that case.
    """
    import json
    import time

    import numpy as np
    import SimpleITK as sitk
    from tqdm import tqdm

    from zonal_reliability_fusion.anatomy.contracts import anatomy_geometry, anatomy_same_grid
    from zonal_reliability_fusion.anatomy.inference import load_anatomy_probability_case

    start = time.monotonic()
    source, priors, model, output = map(Path, (source_raw, prior_dir, model_folder, output_raw))
    if output.exists() or output.is_symlink() or output.name != PREDICTED_DATASET:
        raise ValueError("output must be a new Dataset608 directory; no resume/overwrite")
    if source.name != "Dataset605_PICAI":
        raise ValueError("source must be Dataset605_PICAI")
    splits = json.loads(Path(split_file).read_text())
    split = splits[0]
    anatomy_doc = json.loads((model / "dataset.json").read_text())
    plans = json.loads((model / "plans.json").read_text())
    checkpoint = model / "fold_0/checkpoint_final.pth"
    if not checkpoint.is_file() or plans.get("dataset_name") != "Dataset607_PICAI_Anatomy":
        raise ValueError("missing frozen Stage-1 final checkpoint or wrong model dataset")
    trained = sorted({cid.split("_")[0]
                      for cid in anatomy_doc["anatomy_contract"]["train_cases"]})
    document = {"channel_names": {f"{i:04d}": name for i, name in enumerate(
        ["T2W", "ADC", "HBV", "noNorm", "noNorm", "noNorm"])},
        "labels": {"background": 0, "lesion": 1}, "file_ending": ".nii.gz",
        "numTraining": len(split["train"]) + len(split["val"]),
        "predicted_anatomy_contract": {
            "channel_order": ["WG", "PZ", "TZ"], "source": "predicted_prior",
            "train_cases": split["train"], "val_cases": split["val"], "cases": {},
            "source_lesion_split": str(split_file), "source_anatomy_model": str(model),
        }}
    records = document["predicted_anatomy_contract"]["cases"]
    # Complete preflight before creating the output directory; no silent exclusion.
    for cid in split["train"] + split["val"]:
        records[cid] = {
            "case_id": cid, "checkpoint": str(checkpoint.resolve()),
            "trainer": model.name.split("__")[0], "source_split": "fold_0",
            "prior_mode": "IN_SAMPLE_PRED" if cid.split("_")[0] in trained else "HELD_OUT_PRED",
            "anatomy_training_patient_ids": trained, "geometry": "pending",
        }
        for path in [*(source / "imagesTr" / f"{cid}_{i:04d}.nii.gz" for i in range(3)),
                     source / "labelsTr" / f"{cid}.nii.gz",
                     priors / f"{cid}.npz", priors / f"{cid}.pkl", priors / f"{cid}.nii.gz"]:
            if not path.is_file():
                raise ValueError(f"missing input {path}")
    validate_predicted_prior_dataset(document, output.name)
    output.mkdir(parents=True)
    (output / "imagesTr").mkdir(); (output / "labelsTr").mkdir()
    ok = 0
    try:
        for cid in tqdm(records, desc="predicted anatomy dataset", unit="case", disable=no_progress):
            mri_paths = [source / "imagesTr" / f"{cid}_{i:04d}.nii.gz" for i in range(3)]
            t2 = sitk.ReadImage(str(mri_paths[0]))
            for path in [*mri_paths[1:], source / "labelsTr" / f"{cid}.nii.gz"]:
                anatomy_same_grid(t2, sitk.ReadImage(str(path)))
            probabilities, _, _ = load_anatomy_probability_case(priors, cid, t2, plans["transpose_forward"])
            records[cid]["geometry"] = anatomy_geometry(t2)
            for i, path in enumerate(mri_paths):
                (output / "imagesTr" / f"{cid}_{i:04d}.nii.gz").symlink_to(path.resolve())
            (output / "labelsTr" / f"{cid}.nii.gz").symlink_to(
                (source / "labelsTr" / f"{cid}.nii.gz").resolve())
            for index, probability in enumerate(probabilities):
                image = sitk.GetImageFromArray(probability.astype(np.float32))
                image.CopyInformation(t2)
                sitk.WriteImage(image, str(output / "imagesTr" / f"{cid}_{index+3:04d}.nii.gz"))
            ok += 1
        validate_predicted_prior_dataset(document, output.name)
        (output / "dataset.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        (output / "splits_final.json").write_text(json.dumps(splits, indent=2) + "\n")
    except Exception:
        print(f"[predicted-dataset] success={ok} failed=1 skipped=0 "
              f"elapsed={time.monotonic()-start:.2f}s output={output} INCOMPLETE; preserved")
        raise
    print(f"[predicted-dataset] success={ok} failed=0 skipped=0 "
          f"elapsed={time.monotonic()-start:.2f}s output={output}")
    return document


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Materialize new predicted-prior Dataset608; user-run")
    for key in ("source-raw", "prior-dir", "model-folder", "output-raw", "split-file"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--no-progress", action="store_true")
    materialize_predicted_dataset(**vars(parser.parse_args()))
