"""Prostate158 独立外测准备与评估的纯合成 CPU 测试。

全部使用 ``tmp_path`` 中自造的小 CSV 与合成 NIfTI（SimpleITK），CPU 运行；不读取真实
Prostate158 影像，不写 data/ workdir/ outputs/，不启动推理 / 训练，不使用 GPU。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "prepare_prostate158_external.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_external_segmentation.py"

IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
ROT180_Z_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
CSV_HEADER = (
    "ID,t2,adc,dwi,t2_anatomy_reader1,t2_tumor_reader1,adc_tumor_reader1,"
    "t2_anatomy_reader2,adc_tumor_reader2"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = _load(PREPARE_SCRIPT, "prep_prostate158_test_module")
EVAL = _load(EVAL_SCRIPT, "eval_prostate158_test_module")


def _write_img(
    path,
    arr,
    *,
    size_xyz=None,
    spacing_xyz=(1.0, 1.0, 2.0),
    origin_xyz=(0.0, 0.0, 0.0),
    direction=IDENTITY_DIRECTION,
) -> None:
    import SimpleITK as sitk

    if size_xyz is not None:
        assert tuple(arr.shape) == (size_xyz[2], size_xyz[1], size_xyz[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetOrigin(tuple(float(o) for o in origin_xyz))
    img.SetDirection(tuple(float(d) for d in direction))
    sitk.WriteImage(img, str(path), useCompression=True)


def _intensity(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 200, size=(4, 8, 8), dtype=np.int16)


def _box_mask(size_xyz, spacing_xyz, origin_xyz, direction, box) -> np.ndarray:
    """在给定网格上按物理坐标半开区间生成掩膜（z,y,x 数组）。"""
    nx, ny, nz = size_xyz
    rot = np.asarray(direction, dtype=float).reshape(3, 3)
    sp = np.asarray(spacing_xyz, dtype=float)
    origin = np.asarray(origin_xyz, dtype=float)
    arr = np.zeros((nz, ny, nx), dtype=np.uint8)
    x0, x1, y0, y1, z0, z1 = box
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                p = origin + rot @ (np.array([ix, iy, iz], dtype=float) * sp)
                if x0 <= p[0] < x1 and y0 <= p[1] < y1 and z0 <= p[2] < z1:
                    arr[iz, iy, ix] = 1
    return arr


# T2 参考网格：size (8,8,4)、spacing (1,1,2)、origin 0；物理范围 x/y 0..7、z 0..6
T2_GRID = {
    "size_xyz": (8, 8, 4),
    "spacing_xyz": (1.0, 1.0, 2.0),
    "origin_xyz": (0.0, 0.0, 0.0),
}
# 更细网格：spacing 为 T2 的整数分之一、origin 重合、角点精确落在偶数索引上，
# 因此 NN 重采样在角点采样约定下可精确往返；物理范围覆盖 T2（x/y 0..7、z 0..6）
FINE_GRID = {
    "size_xyz": (15, 15, 7),
    "spacing_xyz": (0.5, 0.5, 1.0),
    "origin_xyz": (0.0, 0.0, 0.0),
}
LESION_BOX = (2.0, 4.0, 2.0, 4.0, 0.0, 2.0)


def _write_case(
    root,
    rel_dir,
    original_id,
    *,
    adc_grid=None,
    dwi_grid=None,
    adc_direction=IDENTITY_DIRECTION,
    primary="positive",
    reader2="positive",
) -> dict:
    """在 root/<rel_dir>/<id>/ 写一例合成影像，返回 CSV 行 dict。"""
    # rel_dir 已含病例目录（与 CSV 行一致，如 test/001），不再追加 original_id
    cdir = root / rel_dir
    _write_img(cdir / "t2.nii.gz", _intensity(int(original_id) + 1), **T2_GRID)
    adc_grid = adc_grid or T2_GRID
    dwi_grid = dwi_grid or T2_GRID
    adc_kw = dict(adc_grid)
    adc_kw.setdefault("direction", adc_direction)
    rng_a = np.random.default_rng(int(original_id) + 2)
    rng_d = np.random.default_rng(int(original_id) + 3)
    adc_arr = rng_a.integers(
        0,
        200,
        size=(
            adc_grid["size_xyz"][2],
            adc_grid["size_xyz"][1],
            adc_grid["size_xyz"][0],
        ),
        dtype=np.int16,
    )
    dwi_arr = rng_d.integers(
        0,
        200,
        size=(
            dwi_grid["size_xyz"][2],
            dwi_grid["size_xyz"][1],
            dwi_grid["size_xyz"][0],
        ),
        dtype=np.int16,
    )
    _write_img(cdir / "adc.nii.gz", adc_arr, **adc_kw)
    _write_img(cdir / "dwi.nii.gz", dwi_arr, **dwi_grid)
    _write_img(
        cdir / "t2_anatomy_reader1.nii.gz",
        np.zeros((4, 8, 8), dtype=np.uint8),
        **T2_GRID,
    )

    if primary == "positive":
        mask = _box_mask(
            T2_GRID["size_xyz"],
            T2_GRID["spacing_xyz"],
            T2_GRID["origin_xyz"],
            IDENTITY_DIRECTION,
            LESION_BOX,
        )
        _write_img(cdir / "adc_tumor_reader1.nii.gz", mask, **T2_GRID)
        _write_img(cdir / "t2_tumor_reader1.nii.gz", mask, **T2_GRID)
        t2_tumor = f"{rel_dir}/t2_tumor_reader1.nii.gz"
        primary_rel = f"{rel_dir}/adc_tumor_reader1.nii.gz"
    elif primary == "negative":
        _write_img(
            cdir / "empty.nii.gz", np.zeros((4, 8, 8), dtype=np.uint8), **T2_GRID
        )
        t2_tumor = ""
        primary_rel = f"{rel_dir}/empty.nii.gz"
    else:
        raise ValueError(primary)

    if reader2 == "positive":
        r2mask = _box_mask(
            T2_GRID["size_xyz"],
            T2_GRID["spacing_xyz"],
            T2_GRID["origin_xyz"],
            IDENTITY_DIRECTION,
            (3.0, 5.0, 3.0, 5.0, 0.0, 2.0),
        )
        _write_img(cdir / "adc_tumor_reader2.nii.gz", r2mask, **T2_GRID)
        r2_rel = f"{rel_dir}/adc_tumor_reader2.nii.gz"
    elif reader2 == "negative":
        _write_img(
            cdir / "empty_r2.nii.gz", np.zeros((4, 8, 8), dtype=np.uint8), **T2_GRID
        )
        r2_rel = f"{rel_dir}/empty_r2.nii.gz"
    else:
        r2_rel = ""

    return {
        "ID": original_id,
        "t2": f"{rel_dir}/t2.nii.gz",
        "adc": f"{rel_dir}/adc.nii.gz",
        "dwi": f"{rel_dir}/dwi.nii.gz",
        "t2_anatomy_reader1": f"{rel_dir}/t2_anatomy_reader1.nii.gz",
        "t2_tumor_reader1": t2_tumor,
        "adc_tumor_reader1": primary_rel,
        "t2_anatomy_reader2": "",
        "adc_tumor_reader2": r2_rel,
    }


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    cols = CSV_HEADER.split(",")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_dataset(tmp_path: Path, queue: str, rows: list[dict]) -> Path:
    if queue == "test":
        csv_path = tmp_path / "prostate158_test" / "test.csv"
    else:
        csv_path = tmp_path / "prostate158_train" / f"{queue}.csv"
    _write_csv(csv_path, rows)
    return csv_path


def _run(module, argv: list[str]) -> int:
    return module.main(argv)


@pytest.fixture()
def one_positive(tmp_path: Path) -> Path:
    row = _write_case(tmp_path / "prostate158_test", "test/001", "001")
    return _make_dataset(tmp_path, "test", [row])


# --------------------------------------------------------------------------- audit / prepare
def test_audit_linkable_positive_and_negative(tmp_path):
    pos = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", primary="positive"
    )
    neg = _write_case(
        tmp_path / "prostate158_test",
        "test/002",
        "002",
        primary="negative",
        reader2="absent",
    )
    csv_path = _make_dataset(tmp_path, "test", [pos, neg])

    records, errors = PREP.parse_csv(csv_path, "test", csv_path.parent)
    assert errors == []
    audited = PREP.run_audit(records, progress=False)
    by_id = {r["case_id"]: r for r in audited}
    assert by_id["P158_test_001"]["category"] == "linkable"
    assert by_id["P158_test_001"]["primary_reference"]["status"] == "positive"
    assert by_id["P158_test_002"]["category"] == "linkable"
    assert by_id["P158_test_002"]["primary_reference"]["status"] == "negative"
    assert by_id["P158_test_002"]["primary_reference"]["explicit_empty_mask"] is True
    payload = PREP._audit_payload(csv_path, "test", csv_path.parent, audited, [], 0.0)
    assert "HBV" in payload["domain_caveat"] and "跨域" in payload["domain_caveat"]
    assert payload["counts"]["primary_positive"] == 1
    assert payload["counts"]["primary_negative_verified_empty"] == 1


def test_prepare_symlinks_channel_order_targets_and_manifest(tmp_path, one_positive):
    out = tmp_path / "ext"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(one_positive),
                "--queue",
                "test",
                "--output-dir",
                str(out),
                "--no-progress",
            ],
        )
        == 0
    )
    images = out / "images"
    cid = "P158_test_001"
    mapping = {
        f"{cid}_0000.nii.gz": "t2.nii.gz",
        f"{cid}_0001.nii.gz": "adc.nii.gz",
        f"{cid}_0002.nii.gz": "dwi.nii.gz",
    }
    for link_name, target_name in mapping.items():
        link = images / link_name
        assert link.is_symlink()
        assert link.readlink()  # 软链接可解析
        assert Path(link.resolve()).name == target_name
    assert not (images / f"{cid}.nii.gz").exists()  # 参考标签不进输入目录
    manifest = json.loads((out / PREP.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["queue"] == "test"
    case = manifest["cases"][0]
    assert case["case_id"] == cid and case["original_id"] == "001"
    assert case["channels"]["t2"]["kind"] == "symlink"
    assert case["primary_reference"]["path"].endswith("adc_tumor_reader1.nii.gz")
    assert case["csv_row_number"] == 2  # 可反查 CSV 行


def test_empty_primary_field_is_failure_not_negative(tmp_path):
    row = _write_case(tmp_path / "prostate158_test", "test/001", "001")
    row["adc_tumor_reader1"] = ""
    csv_path = _make_dataset(tmp_path, "test", [row])
    records, errors = PREP.parse_csv(csv_path, "test", csv_path.parent)
    assert any("空字段不是阴性" in e for e in errors)
    audited = PREP.run_audit(records, progress=False)
    assert audited[0]["category"] == "failed"
    assert (
        _run(
            PREP, ["audit", "--csv", str(csv_path), "--queue", "test", "--no-progress"]
        )
        == 2
    )
    out = tmp_path / "ext"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(csv_path),
                "--queue",
                "test",
                "--output-dir",
                str(out),
                "--no-progress",
            ],
        )
        == 1
    )
    assert not out.exists()


def test_path_traversal_absolute_and_duplicate_ids_rejected(tmp_path):
    row = _write_case(tmp_path / "prostate158_test", "test/001", "001")
    row_abs = dict(row)
    row_abs["t2"] = "/etc/passwd"
    row_dot = dict(row)
    row_dot["ID"] = "002"
    row_dot["adc"] = "../../../../etc/hostname"
    row_dup = dict(row)
    csv_path = _make_dataset(tmp_path, "test", [row_abs, row_dot, row_dup])
    records, errors = PREP.parse_csv(csv_path, "test", csv_path.parent)
    joined = "\n".join(errors)
    assert "不允许是绝对路径" in joined
    assert "越界" in joined
    assert "原始 ID 重复" in joined
    assert all(
        r["category"] == "failed" for r in PREP.run_audit(records, progress=False)
    )


def test_missing_channel_and_missing_label_files_fail(tmp_path):
    row = _write_case(tmp_path / "prostate158_test", "test/001", "001")
    (tmp_path / "prostate158_test/test/001/adc.nii.gz").unlink()
    (tmp_path / "prostate158_test/test/001/adc_tumor_reader1.nii.gz").unlink()
    csv_path = _make_dataset(tmp_path, "test", [row])
    audited = PREP.run_audit(
        PREP.parse_csv(csv_path, "test", csv_path.parent)[0], progress=False
    )
    errs = " ".join(audited[0]["errors"])
    assert "adc 文件不存在" in errs
    assert "主参考标签无效" in errs
    assert audited[0]["category"] == "failed"


def test_same_size_different_origin_is_grid_mismatch(tmp_path):
    shifted = dict(T2_GRID, origin_xyz=(5.0, 0.0, 0.0))  # size 同、origin 不同
    row = _write_case(
        tmp_path / "prostate158_test",
        "test/001",
        "001",
        adc_grid=shifted,
        dwi_grid=shifted,
    )
    csv_path = _make_dataset(tmp_path, "test", [row])
    audited = PREP.run_audit(
        PREP.parse_csv(csv_path, "test", csv_path.parent)[0], progress=False
    )
    r = audited[0]
    assert r["category"] == "grid_mismatch"
    assert "origin" in r["channels_mismatch_fields"]["adc"]
    # 平移后源 FOV（x 5..12）不覆盖 T2（0..7），不可重采样
    assert r["resample_candidate"] is False
    out = tmp_path / "ext"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(csv_path),
                "--queue",
                "test",
                "--output-dir",
                str(out),
                "--no-progress",
            ],
        )
        == 1
    )
    assert not out.exists()


def test_grid_mismatch_skipped_by_default_alongside_linkable(tmp_path):
    good = _write_case(tmp_path / "prostate158_test", "test/001", "001")
    shifted = dict(T2_GRID, origin_xyz=(-3.0, 0.0, 0.0))  # 覆盖不全
    bad = _write_case(
        tmp_path / "prostate158_test",
        "test/002",
        "002",
        adc_grid=shifted,
        dwi_grid=shifted,
    )
    csv_path = _make_dataset(tmp_path, "test", [good, bad])
    out = tmp_path / "ext"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(csv_path),
                "--queue",
                "test",
                "--output-dir",
                str(out),
                "--no-progress",
            ],
        )
        == 0
    )
    assert (out / "images" / "P158_test_001_0000.nii.gz").is_symlink()
    assert not (out / "images" / "P158_test_002_0000.nii.gz").exists()
    manifest = json.loads((out / PREP.MANIFEST_NAME).read_text())
    assert len(manifest["cases"]) == 1
    assert manifest["skipped_cases"][0]["case_id"] == "P158_test_002"


def test_resample_candidate_derived_only_with_explicit_flag(tmp_path):
    row = _write_case(
        tmp_path / "prostate158_test",
        "test/001",
        "001",
        adc_grid=FINE_GRID,
        dwi_grid=FINE_GRID,
    )
    csv_path = _make_dataset(tmp_path, "test", [row])
    audited = PREP.run_audit(
        PREP.parse_csv(csv_path, "test", csv_path.parent)[0], progress=False
    )
    r = audited[0]
    assert r["category"] == "grid_mismatch"
    assert r["resample_candidate"] is True

    # 默认：唯一病例被跳过 -> 无病例可物化 -> 失败且无目录
    out1 = tmp_path / "ext_default"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(csv_path),
                "--queue",
                "test",
                "--output-dir",
                str(out1),
                "--no-progress",
            ],
        )
        == 1
    )
    assert not out1.exists()

    # 显式 --allow-resample：派生文件落在 T2 网格
    import SimpleITK as sitk

    out2 = tmp_path / "ext_resample"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(csv_path),
                "--queue",
                "test",
                "--output-dir",
                str(out2),
                "--allow-resample",
                "--no-progress",
            ],
        )
        == 0
    )
    t2_link = out2 / "images" / "P158_test_001_0000.nii.gz"
    adc_file = out2 / "images" / "P158_test_001_0001.nii.gz"
    assert t2_link.is_symlink()
    assert adc_file.is_file() and not adc_file.is_symlink()
    g = PREP.geometry_of(sitk.ReadImage(str(adc_file)))
    t2g = PREP.geometry_of(sitk.ReadImage(str(t2_link)))
    assert g == t2g
    manifest = json.loads((out2 / PREP.MANIFEST_NAME).read_text())
    assert manifest["cases"][0]["channels"]["adc"]["kind"] == "derived_resampled"
    assert manifest["cases"][0]["channels"]["adc"]["resample"][
        "interpolator"
    ].startswith("SimpleITK.sitkLinear")
    # 原始 adc 仍是细网格，未被修改
    assert PREP.geometry_of(
        sitk.ReadImage(str(csv_path.parent / "test/001/adc.nii.gz"))
    )["size"] == (15, 15, 7)


def test_unresamplable_direction_fail_closed_and_no_staging(tmp_path):
    row = _write_case(
        tmp_path / "prostate158_test",
        "test/001",
        "001",
        adc_grid=T2_GRID,
        dwi_grid=T2_GRID,
        adc_direction=ROT180_Z_DIRECTION,
    )
    csv_path = _make_dataset(tmp_path, "test", [row])
    out = tmp_path / "ext"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(csv_path),
                "--queue",
                "test",
                "--output-dir",
                str(out),
                "--allow-resample",
                "--no-progress",
            ],
        )
        == 1
    )
    assert not out.exists()
    assert not list(tmp_path.glob(".ext.staging-*"))


def test_output_dir_refuses_existing_and_inputs_unchanged(tmp_path, one_positive):
    import hashlib

    out = tmp_path / "ext"
    out.mkdir()
    (out / "sentinel").write_text("x")
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(one_positive),
                "--queue",
                "test",
                "--output-dir",
                str(out),
                "--no-progress",
            ],
        )
        == 1
    )
    assert (out / "sentinel").read_text() == "x"

    src = one_positive.parent / "test/001/t2.nii.gz"
    before = (src.stat().st_size, hashlib.sha256(src.read_bytes()).hexdigest())
    ok = tmp_path / "ext2"
    assert (
        _run(
            PREP,
            [
                "prepare",
                "--csv",
                str(one_positive),
                "--queue",
                "test",
                "--output-dir",
                str(ok),
                "--no-progress",
            ],
        )
        == 0
    )
    after = (src.stat().st_size, hashlib.sha256(src.read_bytes()).hexdigest())
    assert before == after


# --------------------------------------------------------------------------- 端到端：prepare → 评估
def _prepared_manifest(
    tmp_path: Path,
    rows: list[dict],
    queue: str = "test",
    out_name: str = "ext",
    allow_resample: bool = False,
) -> Path:
    if queue == "test":
        csv_path = tmp_path / "prostate158_test" / "test.csv"
    else:
        csv_path = tmp_path / "prostate158_train" / f"{queue}.csv"
    out = tmp_path / out_name
    argv = [
        "prepare",
        "--csv",
        str(csv_path),
        "--queue",
        queue,
        "--output-dir",
        str(out),
        "--no-progress",
    ]
    if allow_resample:
        argv.append("--allow-resample")
    assert _run(PREP, argv) == 0
    return out / PREP.MANIFEST_NAME


def _write_predictions(manifest_path: Path, pred_dir: Path, *, kind: str) -> None:
    """在 T2 网格上按物理盒写出合成预测；kind 决定与主参考的关系。"""
    manifest = json.loads(manifest_path.read_text())
    pred_dir.mkdir(parents=True, exist_ok=True)
    for case in manifest["cases"]:
        cid = case["case_id"]
        ref_status = case["primary_reference"]["status"]
        if ref_status == "negative":
            if kind == "neg_fp":
                box = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
                arr = _box_mask(
                    T2_GRID["size_xyz"],
                    T2_GRID["spacing_xyz"],
                    T2_GRID["origin_xyz"],
                    IDENTITY_DIRECTION,
                    box,
                )
            else:
                arr = np.zeros((4, 8, 8), dtype=np.uint8)
        elif kind in ("perfect", "neg_fp"):
            arr = _box_mask(
                T2_GRID["size_xyz"],
                T2_GRID["spacing_xyz"],
                T2_GRID["origin_xyz"],
                IDENTITY_DIRECTION,
                LESION_BOX,
            )
        elif kind == "partial":
            arr = _box_mask(
                T2_GRID["size_xyz"],
                T2_GRID["spacing_xyz"],
                T2_GRID["origin_xyz"],
                IDENTITY_DIRECTION,
                (2.0, 3.0, 2.0, 4.0, 0.0, 2.0),
            )
        elif kind == "missed":
            arr = np.zeros((4, 8, 8), dtype=np.uint8)
        elif kind == "wrong_location":
            arr = _box_mask(
                T2_GRID["size_xyz"],
                T2_GRID["spacing_xyz"],
                T2_GRID["origin_xyz"],
                IDENTITY_DIRECTION,
                (5.0, 7.0, 5.0, 7.0, 4.0, 6.0),
            )
        else:
            raise ValueError(kind)
        _write_img(pred_dir / f"{cid}.nii.gz", arr, **T2_GRID)


def _eval(manifest, model_arg, out, **extra):
    argv = [
        "--manifest",
        str(manifest),
        "--model",
        model_arg,
        "--output",
        str(out),
        "--bootstrap-resamples",
        "50",
        "--no-progress",
    ]
    for k, v in extra.items():
        argv.extend([f"--{k.replace('_', '-')}", v])
    return _run(EVAL, argv)


def test_evaluate_metrics_perfect_partial_missed_wrong_location_and_negative_fp(
    tmp_path,
):
    pos1 = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="absent"
    )
    neg = _write_case(
        tmp_path / "prostate158_test",
        "test/002",
        "002",
        primary="negative",
        reader2="absent",
    )
    _make_dataset(tmp_path, "test", [pos1, neg])
    manifest = _prepared_manifest(tmp_path, [pos1, neg])

    pred = tmp_path / "pred_perfect"
    _write_predictions(manifest, pred, kind="perfect")
    out = tmp_path / "metrics_perfect.json"
    assert _eval(manifest, f"perfect={pred}", out) == 0
    m = json.loads(out.read_text())["models"]["perfect"]
    assert m["case_counts"]["num_cases"] == 2
    assert m["case_counts"]["num_positive_cases"] == 1
    assert m["case_counts"]["num_negative_cases"] == 1
    assert m["positive_segmentation"]["positive_macro_dice_mean"] == pytest.approx(1.0)
    assert m["negative_false_positive"]["negative_fp_cases"] == 0
    assert m["voxel_metrics"]["all_prediction_voxel_precision"] == pytest.approx(1.0)

    for kind, check in (
        (
            "partial",
            lambda mm: 0 < mm["positive_segmentation"]["positive_macro_dice_mean"] < 1,
        ),
        (
            "missed",
            lambda mm: (
                mm["miss_structure"]["positive_missed_cases"] == 1
                and mm["positive_segmentation"]["positive_macro_dice_mean"] == 0.0
            ),
        ),
        (
            "wrong_location",
            lambda mm: (
                mm["miss_structure"]["positive_wrong_location_cases"] == 1
                and mm["voxel_metrics"]["positive_voxel_recall"] == 0.0
            ),
        ),
    ):
        pdir = tmp_path / f"pred_{kind}"
        _write_predictions(manifest, pdir, kind=kind)
        opath = tmp_path / f"metrics_{kind}.json"
        assert _eval(manifest, f"{kind}={pdir}", opath) == 0
        assert check(json.loads(opath.read_text())["models"][kind])

    pdir = tmp_path / "pred_negfp"
    _write_predictions(manifest, pdir, kind="neg_fp")
    opath = tmp_path / "metrics_negfp.json"
    assert _eval(manifest, f"negfp={pdir}", opath) == 0
    mm = json.loads(opath.read_text())["models"]["negfp"]
    assert mm["negative_false_positive"]["negative_fp_cases"] == 1
    assert mm["negative_false_positive"]["negative_fp_voxels_total"] > 0
    assert 0 < mm["voxel_metrics"]["all_prediction_voxel_precision"] < 1


def test_evaluate_no_negatives_marks_not_applicable_not_zero(tmp_path):
    pos = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="absent"
    )
    _make_dataset(tmp_path, "test", [pos])
    manifest = _prepared_manifest(tmp_path, [pos])
    pred = tmp_path / "pred"
    _write_predictions(manifest, pred, kind="perfect")
    out = tmp_path / "metrics.json"
    assert _eval(manifest, f"m={pred}", out) == 0
    m = json.loads(out.read_text())["models"]["m"]
    assert m["negative_false_positive"]["applicable"] is False
    assert m["negative_false_positive"]["negative_fp_cases"] is None
    assert m["voxel_metrics"]["all_prediction_voxel_precision"] is None


def test_evaluate_nearest_neighbor_alignment_when_grids_differ(tmp_path):
    pos = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="absent"
    )
    _make_dataset(tmp_path, "test", [pos])
    manifest = _prepared_manifest(tmp_path, [pos])
    pred_dir = tmp_path / "pred_fine"
    pred_dir.mkdir()
    fine_mask = _box_mask(
        FINE_GRID["size_xyz"],
        FINE_GRID["spacing_xyz"],
        FINE_GRID["origin_xyz"],
        IDENTITY_DIRECTION,
        LESION_BOX,
    )
    _write_img(pred_dir / "P158_test_001.nii.gz", fine_mask, **FINE_GRID)
    out = tmp_path / "metrics.json"
    assert _eval(manifest, f"m={pred_dir}", out) == 0
    m = json.loads(out.read_text())["models"]["m"]
    assert (
        m["evaluation_grid_alignment"]["cases_with_prediction_to_reference_nn_resample"]
        == 1
    )
    assert m["positive_segmentation"]["positive_macro_dice_mean"] > 0.8

    # 预测 FOV 不覆盖参考网格（x 仅 0..2）→ fail-closed
    bad_geom = {
        "size_xyz": (3, 8, 4),
        "spacing_xyz": (1.0, 1.0, 2.0),
        "origin_xyz": (0.0, 0.0, 0.0),
    }
    bad_dir = tmp_path / "pred_bad"
    bad_dir.mkdir()
    bad_mask = _box_mask(
        bad_geom["size_xyz"],
        bad_geom["spacing_xyz"],
        bad_geom["origin_xyz"],
        IDENTITY_DIRECTION,
        (0.0, 2.0, 0.0, 2.0, 0.0, 2.0),
    )
    _write_img(bad_dir / "P158_test_001.nii.gz", bad_mask, **bad_geom)
    out_bad = tmp_path / "metrics_bad.json"
    assert _eval(manifest, f"m={bad_dir}", out_bad) == 1
    assert not out_bad.exists()


def test_evaluate_different_direction_frame_fails(tmp_path):
    pos = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="absent"
    )
    _make_dataset(tmp_path, "test", [pos])
    manifest = _prepared_manifest(tmp_path, [pos])
    pred_dir = tmp_path / "pred_rot"
    pred_dir.mkdir()
    arr = _box_mask(
        T2_GRID["size_xyz"],
        T2_GRID["spacing_xyz"],
        T2_GRID["origin_xyz"],
        ROT180_Z_DIRECTION,
        (-7.0, -5.0, 2.0, 4.0, 0.0, 2.0),
    )
    _write_img(
        pred_dir / "P158_test_001.nii.gz",
        arr,
        **{**T2_GRID, "direction": ROT180_Z_DIRECTION},
    )
    out = tmp_path / "metrics.json"
    assert _eval(manifest, f"m={pred_dir}", out) == 1
    assert not out.exists()


def test_evaluate_non_binary_label_fails_and_does_not_publish(tmp_path):
    pos = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="absent"
    )
    _make_dataset(tmp_path, "test", [pos])
    manifest = _prepared_manifest(tmp_path, [pos])
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    _write_img(
        pred_dir / "P158_test_001.nii.gz",
        np.full((4, 8, 8), 2, dtype=np.uint8),
        **T2_GRID,
    )
    out = tmp_path / "metrics.json"
    assert _eval(manifest, f"m={pred_dir}", out) == 1
    assert not out.exists()


def test_evaluate_missing_prediction_fails(tmp_path):
    pos = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="absent"
    )
    _make_dataset(tmp_path, "test", [pos])
    manifest = _prepared_manifest(tmp_path, [pos])
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    out = tmp_path / "metrics.json"
    assert _eval(manifest, f"m={pred_dir}", out) == 1
    assert not out.exists()


def test_multi_model_comparability_and_paired(tmp_path):
    rows = [
        _write_case(tmp_path / "prostate158_test", "test/001", "001", reader2="absent"),
        _write_case(
            tmp_path / "prostate158_test",
            "test/002",
            "002",
            primary="negative",
            reader2="absent",
        ),
    ]
    _make_dataset(tmp_path, "test", rows)
    manifest = _prepared_manifest(tmp_path, rows)
    p_good = tmp_path / "pred_good"
    p_bad = tmp_path / "pred_bad"
    _write_predictions(manifest, p_good, kind="perfect")
    _write_predictions(manifest, p_bad, kind="partial")
    out = tmp_path / "metrics_pair.json"
    argv = [
        "--manifest",
        str(manifest),
        "--model",
        f"good={p_good}",
        "--model",
        f"bad={p_bad}",
        "--output",
        str(out),
        "--bootstrap-resamples",
        "100",
        "--no-progress",
    ]
    assert _run(EVAL, argv) == 0
    doc = json.loads(out.read_text())
    assert len(doc["paired_comparisons"]) == 1
    cmp_ = doc["paired_comparisons"][0]
    assert cmp_["model_a"] == "good" and cmp_["model_b"] == "bad"
    assert cmp_["mean_paired_dice_delta"] < 0

    # 从 good 目录移走一个病例 -> 病例集合不一致 -> 拒绝比较、不发布
    p_inc = tmp_path / "pred_inc"
    p_inc.mkdir()
    (p_good / "P158_test_001.nii.gz").replace(p_inc / "P158_test_001.nii.gz")
    out2 = tmp_path / "metrics_incompatible.json"
    argv2 = [
        "--manifest",
        str(manifest),
        "--model",
        f"good={p_good}",
        "--model",
        f"bad={p_bad}",
        "--output",
        str(out2),
        "--bootstrap-resamples",
        "50",
        "--no-progress",
    ]
    assert _run(EVAL, argv2) == 1
    assert not out2.exists()


def test_reader2_subset_only_and_no_fallback(tmp_path):
    with_r2 = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="positive"
    )
    no_r2 = _write_case(
        tmp_path / "prostate158_test", "test/002", "002", reader2="absent"
    )
    _make_dataset(tmp_path, "test", [with_r2, no_r2])
    manifest = _prepared_manifest(tmp_path, [with_r2, no_r2])

    pred = tmp_path / "pred_r2"
    pred.mkdir()
    r2_box = (3.0, 5.0, 3.0, 5.0, 0.0, 2.0)
    _write_img(
        pred / "P158_test_001.nii.gz",
        _box_mask(
            T2_GRID["size_xyz"],
            T2_GRID["spacing_xyz"],
            T2_GRID["origin_xyz"],
            IDENTITY_DIRECTION,
            r2_box,
        ),
        **T2_GRID,
    )
    out = tmp_path / "metrics_r2.json"
    assert _eval(manifest, f"m={pred}", out, reader="reader2") == 0
    doc = json.loads(out.read_text())
    assert doc["reader"] == "reader2"
    assert doc["num_selected_cases"] == 1
    assert doc["models"]["m"]["case_counts"]["num_cases"] == 1

    (pred / "P158_test_001.nii.gz").unlink()
    out2 = tmp_path / "metrics_r2_missing.json"
    assert _eval(manifest, f"m={pred}", out2, reader="reader2") == 1
    assert not out2.exists()


def test_train_and_valid_queues_have_distinct_case_ids(tmp_path):
    row_t = _write_case(
        tmp_path / "prostate158_train", "train/024", "024", primary="negative"
    )
    row_v = _write_case(
        tmp_path / "prostate158_train", "train/020", "020", primary="positive"
    )
    root = tmp_path / "prostate158_train"
    _write_csv(root / "train.csv", [row_t])
    _write_csv(root / "valid.csv", [row_v])
    for queue, cid, status in (
        ("train", "P158_train_024", "negative"),
        ("valid", "P158_valid_020", "positive"),
    ):
        out = tmp_path / f"ext_{queue}"
        assert (
            _run(
                PREP,
                [
                    "prepare",
                    "--csv",
                    str(root / f"{queue}.csv"),
                    "--queue",
                    queue,
                    "--output-dir",
                    str(out),
                    "--no-progress",
                ],
            )
            == 0
        )
        manifest = json.loads((out / PREP.MANIFEST_NAME).read_text())
        assert manifest["queue"] == queue
        assert manifest["cases"][0]["case_id"] == cid
        assert manifest["cases"][0]["primary_reference"]["status"] == status


def test_reference_file_never_modified_by_evaluation(tmp_path):
    import hashlib

    pos = _write_case(
        tmp_path / "prostate158_test", "test/001", "001", reader2="absent"
    )
    _make_dataset(tmp_path, "test", [pos])
    manifest = _prepared_manifest(tmp_path, [pos])
    ref = tmp_path / "prostate158_test/test/001/adc_tumor_reader1.nii.gz"
    before = hashlib.sha256(ref.read_bytes()).hexdigest()
    pred = tmp_path / "pred"
    _write_predictions(manifest, pred, kind="perfect")
    out = tmp_path / "metrics.json"
    assert _eval(manifest, f"m={pred}", out) == 0
    assert hashlib.sha256(ref.read_bytes()).hexdigest() == before
