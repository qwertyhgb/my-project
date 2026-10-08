"""prepare_picai_nnunet.py 的 fail-closed 行为测试（临时目录 + 纯合成小文件，不读真实医学数据）。

覆盖：任一 failed/conflict 时先打印汇总再非零退出且不发布 dataset.json；numTraining 与磁盘完整
病例集合一致；--overwrite + --cases 子集运行不静默遗留旧病例；splits_final.json 冲突非零退出；
zonal 5 通道 dataset.json（PZ/TZ=noNorm）与几何不一致时 fail-closed。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "prepare_picai_nnunet.py"


def _load_prepare():
    spec = importlib.util.spec_from_file_location(
        "prepare_picai_nnunet", PREPARE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, cids: list[str]) -> None:
    rows = []
    for cid in cids:
        pid, sid = cid.split("_")
        rows.append(
            {
                "case_id": cid,
                "study_id": sid,
                "patient_id": pid,
                "center": "c0",
                "case_csPCa": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_case(
    mat_root: Path,
    cid: str,
    *,
    files=("t2w.nii.gz", "adc.nii.gz", "hbv.nii.gz", "lesion.nii.gz"),
) -> Path:
    case_dir = mat_root / "cases" / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (case_dir / name).write_bytes(b"x")  # 非空占位；baseline 只建符号链接，不读体素
    return case_dir


def _run(mod, argv):
    args = mod.build_parser().parse_args(argv)
    args.func(args)


# --------------------------------------------------------------------------- baseline
def test_baseline_success_writes_dataset_json_with_correct_numtraining(tmp_path):
    mod = _load_prepare()
    mat = tmp_path / "mat"
    raw = tmp_path / "raw"
    cids = ["1_10", "2_20"]
    for cid in cids:
        _make_case(mat, cid)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, cids)

    _run(
        mod,
        [
            "baseline",
            "--materialized-root",
            str(mat),
            "--manifest",
            str(manifest),
            "--nnunet-raw-root",
            str(raw),
            "--no-progress",
        ],
    )

    ds_dir = raw / "Dataset605_PICAI"
    doc = json.loads((ds_dir / "dataset.json").read_text())
    assert doc["numTraining"] == 2
    assert doc["channel_names"] == {"0000": "T2W", "0001": "ADC", "0002": "HBV"}
    assert (ds_dir / "imagesTr" / "1_10_0000.nii.gz").exists()
    assert (ds_dir / "labelsTr" / "1_10.nii.gz").exists()


def test_baseline_fail_closed_on_missing_source_writes_no_dataset_json(
    tmp_path, capsys
):
    mod = _load_prepare()
    mat = tmp_path / "mat"
    raw = tmp_path / "raw"
    _make_case(mat, "1_10")
    _make_case(
        mat, "2_20", files=("t2w.nii.gz", "adc.nii.gz", "lesion.nii.gz")
    )  # 缺 hbv
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["1_10", "2_20"])

    with pytest.raises(SystemExit) as exc:
        _run(
            mod,
            [
                "baseline",
                "--materialized-root",
                str(mat),
                "--manifest",
                str(manifest),
                "--nnunet-raw-root",
                str(raw),
                "--no-progress",
            ],
        )
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "fail-closed" in out  # 先打印完整汇总
    assert not (
        raw / "Dataset605_PICAI" / "dataset.json"
    ).exists()  # 不发布 dataset.json


def test_subset_overwrite_does_not_silently_leave_old_cases(tmp_path, capsys):
    mod = _load_prepare()
    mat = tmp_path / "mat"
    raw = tmp_path / "raw"
    cids = ["1_10", "2_20"]
    for cid in cids:
        _make_case(mat, cid)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, cids)
    base = [
        "baseline",
        "--materialized-root",
        str(mat),
        "--manifest",
        str(manifest),
        "--nnunet-raw-root",
        str(raw),
        "--no-progress",
    ]
    _run(mod, base)  # 完整运行：2 例

    capsys.readouterr()  # 清空
    _run(mod, base + ["--cases", "1_10", "--overwrite"])  # 子集 + overwrite
    out = capsys.readouterr().out
    assert "WARN" in out and "2_20" in out  # 明确提示遗留旧病例，绝不静默
    doc = json.loads((raw / "Dataset605_PICAI" / "dataset.json").read_text())
    assert doc["numTraining"] == 2  # 与磁盘实际完整病例集合一致（不是子集的 1）


# --------------------------------------------------------------------------- splits
def test_splits_conflict_exits_nonzero(tmp_path):
    mod = _load_prepare()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["1_10", "2_20"])
    split_json = tmp_path / "split.json"
    split_json.write_text(
        json.dumps(
            {
                "unit": "study",
                "splits": {
                    "train": {
                        "studies": ["10"],
                        "patients": ["1"],
                        "n_studies": 1,
                        "n_patients": 1,
                    },
                    "validation": {
                        "studies": ["20"],
                        "patients": ["2"],
                        "n_studies": 1,
                        "n_patients": 1,
                    },
                },
            }
        )
    )
    pre_root = tmp_path / "pre"
    pre_dir = pre_root / "Dataset605_PICAI"
    pre_dir.mkdir(parents=True)
    # 预置一个内容不同的 splits_final.json -> 冲突
    (pre_dir / "splits_final.json").write_text(
        json.dumps([{"train": ["x"], "val": ["y"]}])
    )

    with pytest.raises(SystemExit) as exc:
        _run(
            mod,
            [
                "splits",
                "--split-json",
                str(split_json),
                "--manifest",
                str(manifest),
                "--preprocessed-root",
                str(pre_root),
                "--nnunet-raw-root",
                str(tmp_path / "no_raw"),
                "--dataset-id",
                "605",
                "--dataset-name",
                "PICAI",
            ],
        )
    assert exc.value.code == 2
    # 冲突时不得覆盖既有文件
    assert json.loads((pre_dir / "splits_final.json").read_text()) == [
        {"train": ["x"], "val": ["y"]}
    ]


# --------------------------------------------------------------------------- zonal
def _write_nifti(path: Path, arr, spacing=(1.0, 1.0, 1.0)) -> None:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing))
    sitk.WriteImage(img, str(path))


def _make_zonal_case(
    mat_root: Path, cid: str, *, zonal_spacing=(1.0, 1.0, 1.0)
) -> Path:
    import numpy as np

    case_dir = mat_root / "cases" / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    t2w = np.random.default_rng(0).normal(size=(4, 6, 6)).astype("float32")
    _write_nifti(case_dir / "t2w.nii.gz", t2w, spacing=(1.0, 1.0, 1.0))
    label = np.zeros((4, 6, 6), dtype="uint8")
    label[0, 0, 0] = 1  # PZ
    label[1, 1, 1] = 2  # TZ
    _write_nifti(case_dir / "zonal_yuan.nii.gz", label, spacing=zonal_spacing)
    for name in ("adc.nii.gz", "hbv.nii.gz", "lesion.nii.gz"):
        (case_dir / name).write_bytes(b"x")
    return case_dir


def test_zonal_success_five_channels_nonorm(tmp_path):
    mod = _load_prepare()
    mat = tmp_path / "mat"
    raw = tmp_path / "raw"
    _make_zonal_case(mat, "1_10")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["1_10"])

    _run(
        mod,
        [
            "zonal",
            "--zonal-source",
            "yuan",
            "--materialized-root",
            str(mat),
            "--manifest",
            str(manifest),
            "--nnunet-raw-root",
            str(raw),
            "--no-progress",
        ],
    )

    ds_dir = raw / "Dataset606_PICAI_Zonal"
    doc = json.loads((ds_dir / "dataset.json").read_text())
    assert doc["numTraining"] == 1
    assert doc["channel_names"] == {
        "0000": "T2W",
        "0001": "ADC",
        "0002": "HBV",
        "0003": "noNorm",
        "0004": "noNorm",
    }
    # description 术语：算法生成的 zonal membership，而非过强的 fractional occupancy
    desc = doc["description"]
    assert "zonal membership" in desc
    assert "fractional occupancy" not in desc
    assert "[0,1]" in desc and "noNorm" in desc
    assert "not calibrated probabilities" in desc
    assert "prior_source=zonal_yuan" in desc
    assert (ds_dir / "imagesTr" / "1_10_0003.nii.gz").exists()
    assert (ds_dir / "imagesTr" / "1_10_0004.nii.gz").exists()


def test_zonal_geometry_mismatch_fail_closed(tmp_path, capsys):
    mod = _load_prepare()
    mat = tmp_path / "mat"
    raw = tmp_path / "raw"
    _make_zonal_case(mat, "1_10", zonal_spacing=(2.0, 2.0, 2.0))  # 与 T2W 网格不一致
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["1_10"])

    with pytest.raises(SystemExit) as exc:
        _run(
            mod,
            [
                "zonal",
                "--zonal-source",
                "yuan",
                "--materialized-root",
                str(mat),
                "--manifest",
                str(manifest),
                "--nnunet-raw-root",
                str(raw),
                "--no-progress",
            ],
        )
    assert exc.value.code == 2
    assert "fail-closed" in capsys.readouterr().out
    assert not (raw / "Dataset606_PICAI_Zonal" / "dataset.json").exists()


# ------------------------------------------------------------------ zonal 来源安全（必填 + 复用校验）
def _make_case_with_two_sources(mat_root: Path, cid: str) -> Path:
    """写入 t2w + zonal_yuan + zonal_hevi（两者派生的 PZ/TZ 内容不同）。"""
    import numpy as np

    case_dir = mat_root / "cases" / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    t2w = np.random.default_rng(0).normal(size=(4, 6, 6)).astype("float32")
    _write_nifti(case_dir / "t2w.nii.gz", t2w, spacing=(1.0, 1.0, 1.0))
    yuan = np.zeros((4, 6, 6), dtype="uint8")
    yuan[0, 0, 0] = 1  # PZ
    yuan[1, 1, 1] = 2  # TZ
    hevi = np.zeros((4, 6, 6), dtype="uint8")
    hevi[2, 2, 2] = 1  # 与 yuan 不同的 PZ 位置
    _write_nifti(case_dir / "zonal_yuan.nii.gz", yuan)
    _write_nifti(case_dir / "zonal_hevi.nii.gz", hevi)
    for name in ("adc.nii.gz", "hbv.nii.gz", "lesion.nii.gz"):
        (case_dir / name).write_bytes(b"x")
    return case_dir


def _zonal_argv(mat, manifest, raw, source, *extra):
    return [
        "zonal",
        "--zonal-source",
        source,
        "--materialized-root",
        str(mat),
        "--manifest",
        str(manifest),
        "--nnunet-raw-root",
        str(raw),
        "--no-progress",
        *extra,
    ]


def _read_prior(path: Path):
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _zonal_env(tmp_path):
    mat, raw = tmp_path / "mat", tmp_path / "raw"
    _make_case_with_two_sources(mat, "1_10")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["1_10"])
    return mat, raw, manifest, raw / "Dataset606_PICAI_Zonal"


def test_zonal_missing_source_exits_nonzero(tmp_path):
    mod = _load_prepare()
    with pytest.raises(SystemExit) as exc:
        mod.build_parser().parse_args(
            ["zonal", "--nnunet-raw-root", str(tmp_path / "raw")]
        )
    assert exc.value.code == 2  # argparse：缺少必填 --zonal-source


def test_zonal_resume_skips_when_prior_matches_source(tmp_path):
    mod = _load_prepare()
    mat, raw, manifest, ds = _zonal_env(tmp_path)
    _run(mod, _zonal_argv(mat, manifest, raw, "yuan"))  # 首次生成
    out_pz = ds / "imagesTr" / "1_10_0003.nii.gz"
    mtime = out_pz.stat().st_mtime_ns
    _run(mod, _zonal_argv(mat, manifest, raw, "yuan", "--resume"))  # 一致 -> skipped
    assert out_pz.stat().st_mtime_ns == mtime  # 未改写


def test_zonal_source_switch_conflicts_without_overwrite(tmp_path):
    mod = _load_prepare()
    mat, raw, manifest, ds = _zonal_env(tmp_path)
    _run(mod, _zonal_argv(mat, manifest, raw, "yuan"))
    out_pz = ds / "imagesTr" / "1_10_0003.nii.gz"
    pz_before = _read_prior(out_pz).copy()
    dsj_before = (ds / "dataset.json").read_text()
    assert "zonal_yuan" in dsj_before  # 可审计：来源已写入 description
    with pytest.raises(SystemExit) as exc:
        _run(
            mod, _zonal_argv(mat, manifest, raw, "hevi", "--resume")
        )  # 来源切换、内容不同
    assert exc.value.code == 2
    assert _read_prior(out_pz).tolist() == pz_before.tolist()  # 既有 PZ 未修改
    assert (ds / "dataset.json").read_text() == dsj_before  # dataset.json 未更新


def test_zonal_partial_prior_without_overwrite_conflicts(tmp_path):
    mod = _load_prepare()
    mat, raw, manifest, ds = _zonal_env(tmp_path)
    _run(mod, _zonal_argv(mat, manifest, raw, "yuan"))
    out_pz = ds / "imagesTr" / "1_10_0003.nii.gz"
    out_tz = ds / "imagesTr" / "1_10_0004.nii.gz"
    out_tz.unlink()  # 制造“只存在 PZ”的部分状态
    with pytest.raises(SystemExit) as exc:
        _run(mod, _zonal_argv(mat, manifest, raw, "yuan", "--resume"))
    assert exc.value.code == 2
    assert out_pz.exists()  # 已有 PZ 未被删改
    assert not out_tz.exists()  # 缺失的 TZ 未被补写


def test_zonal_overwrite_regenerates_for_new_source(tmp_path):
    import numpy as np

    mod = _load_prepare()
    mat, raw, manifest, ds = _zonal_env(tmp_path)
    _run(mod, _zonal_argv(mat, manifest, raw, "yuan"))
    _run(mod, _zonal_argv(mat, manifest, raw, "hevi", "--overwrite"))  # 显式覆盖
    exp = np.zeros((4, 6, 6), dtype="float32")
    exp[2, 2, 2] = 1.0  # hevi 的 PZ
    got = _read_prior(ds / "imagesTr" / "1_10_0003.nii.gz").astype("float32")
    assert np.array_equal(got, exp)
    desc = json.loads((ds / "dataset.json").read_text())["description"]
    assert "prior_source=zonal_hevi" in desc
    assert "zonal membership" in desc
    assert "fractional occupancy" not in desc


# Anatomy uses only synthetic source files and a scaled frozen metadata fixture.
def _anatomy_fixture(tmp_path, monkeypatch):
    import numpy as np

    from zonal_reliability_fusion.anatomy import contracts as anatomy_contracts
    from zonal_reliability_fusion.nnunet import trainers

    monkeypatch.setattr(anatomy_contracts, "ANATOMY_COUNTS", (2, 1, 1))
    monkeypatch.setattr(trainers, "ANATOMY_COUNTS", (2, 1, 1), raising=False)
    mod = _load_prepare()
    mat, raw = tmp_path / "mat", tmp_path / "raw"
    for cid in ("1_10", "2_20"):
        case = _make_zonal_case(mat, cid)
        wg = np.zeros((4, 6, 6), dtype=np.uint8)
        wg[0, 0, 0] = 1  # PZ inside WG, TZ outside WG; retain disagreement.
        wg[2, 2, 2] = 1  # WG only
        _write_nifti(case / "wg.nii.gz", wg)
    known = trainers.ANATOMY_KNOWN_MISSING
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["1_10", "2_20", known])
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"splits": {"train": {"studies": ["10", "1001070"]}, "validation": {"studies": ["20"]}}}))
    argv = ["anatomy", "--materialized-root", str(mat), "--manifest", str(manifest),
            "--split-json", str(frozen), "--nnunet-raw-root", str(raw),
            "--exclude-known-missing-wg", "--no-progress"]
    return mod, mat, raw, manifest, frozen, argv


def test_anatomy_materialization_contract_and_verified_resume(tmp_path, monkeypatch):
    import SimpleITK as sitk
    mod, mat, raw, _, _, argv = _anatomy_fixture(tmp_path, monkeypatch)
    _run(mod, argv + ["--dry-run"])
    ds = raw / "Dataset607_PICAI_Anatomy"
    assert not ds.exists()
    _run(mod, argv)
    doc = json.loads((ds / "dataset.json").read_text())
    assert list(doc['labels']) == ['background', 'WG', 'PZ', 'TZ']
    assert doc['channel_names'] == {'0000': 'T2W'}
    assert doc['regions_class_order'] == [1, 2, 4]
    assert doc['numTraining'] == 2
    assert doc['anatomy_contract']['explicit_exclusions'] == ['11050_1001070']
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(ds / 'labelsTr/1_10.nii.gz')))
    assert (label[0, 0, 0], label[1, 1, 1], label[2, 2, 2]) == (3, 4, 1)
    before = {p: p.read_bytes() for p in ds.rglob('*') if p.is_file()}
    _run(mod, argv + ['--resume'])
    assert all(p.read_bytes() == contents for p, contents in before.items())
    with pytest.raises(SystemExit):
        _run(mod, argv)
    # Dict equality ignores insertion order; region semantics must not.
    reordered = dict(doc, labels=dict(reversed(list(doc['labels'].items()))))
    (ds / 'dataset.json').write_text(json.dumps(reordered))
    with pytest.raises(SystemExit, match='insertion order conflict'):
        _run(mod, argv + ['--resume'])
    (ds / 'dataset.json').write_bytes(before[ds / 'dataset.json'])
    # Changing supervision/source cannot be silently resumed.
    case = mat / 'cases/1_10'
    changed = sitk.GetArrayFromImage(sitk.ReadImage(str(case / 'wg.nii.gz')))
    changed[3, 3, 3] = 1
    _write_nifti(case / 'wg.nii.gz', changed)
    with pytest.raises(SystemExit):
        _run(mod, argv + ['--resume'])
    assert all(p.read_bytes() == contents for p, contents in before.items() if not p.is_symlink())


@pytest.mark.parametrize('name,value', [('wg.nii.gz', .5), ('wg.nii.gz', 2.),
    ('zonal_yuan.nii.gz', 1.5), ('zonal_yuan.nii.gz', -1.), ('wg.nii.gz', float('nan'))])
def test_anatomy_illegal_labels_fail_before_cast(tmp_path, monkeypatch, name, value):
    import numpy as np
    mod, mat, raw, _, _, argv = _anatomy_fixture(tmp_path, monkeypatch)
    label = np.zeros((4, 6, 6), dtype=np.float32)
    label[0, 0, 0] = value
    _write_nifti(mat / 'cases/1_10' / name, label)
    with pytest.raises(SystemExit):
        _run(mod, argv)
    assert not (raw / 'Dataset607_PICAI_Anatomy/dataset.json').exists()


def test_anatomy_additional_missing_and_explicit_exclusion_required(tmp_path, monkeypatch, capsys):
    mod, mat, raw, _, _, argv = _anatomy_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match='exclude-known-missing'):
        _run(mod, [a for a in argv if a != '--exclude-known-missing-wg'])
    (mat / 'cases/2_20/wg.nii.gz').unlink()  # synthetic only
    with pytest.raises(SystemExit):
        _run(mod, argv)
    assert '2_20' in capsys.readouterr().out
    assert not (raw / 'Dataset607_PICAI_Anatomy/dataset.json').exists()


def test_anatomy_same_shape_different_physical_grid_fails(tmp_path, monkeypatch):
    import SimpleITK as sitk
    mod, mat, raw, _, _, argv = _anatomy_fixture(tmp_path, monkeypatch)
    path = mat / 'cases/1_10/wg.nii.gz'
    image = sitk.ReadImage(str(path))
    image.SetOrigin((1., 0., 0.))
    sitk.WriteImage(image, str(path))
    with pytest.raises(SystemExit):
        _run(mod, argv)
    assert not (raw / 'Dataset607_PICAI_Anatomy/dataset.json').exists()


def test_anatomy_split_exact_scope_patient_isolation_and_no_overwrite(tmp_path, monkeypatch):
    mod, _, raw, manifest, frozen, argv = _anatomy_fixture(tmp_path, monkeypatch)
    _run(mod, argv)
    pre = tmp_path / 'preprocessed'
    (pre / 'Dataset607_PICAI_Anatomy').mkdir(parents=True)
    split_argv = ['splits', '--dataset-id', '607', '--dataset-name', 'PICAI_Anatomy',
                  '--manifest', str(manifest), '--split-json', str(frozen),
                  '--preprocessed-root', str(pre), '--nnunet-raw-root', str(raw), '--exclude-known-missing-wg']
    _run(mod, split_argv)
    path = pre / 'Dataset607_PICAI_Anatomy/splits_final.json'
    assert json.loads(path.read_text()) == [{'train': ['1_10'], 'val': ['2_20']}]
    with pytest.raises(SystemExit):
        _run(mod, split_argv + ['--overwrite'])
    original = path.read_bytes()
    path.write_text('[]')
    with pytest.raises(SystemExit):
        _run(mod, split_argv)
    assert path.read_text() == '[]'
    path.write_bytes(original)
    frozen.write_text(json.dumps({'splits': {'train': {'studies': ['10', '10', '1001070']}, 'validation': {'studies': ['20']}}}))
    with pytest.raises(SystemExit):
        _run(mod, split_argv)


def test_anatomy_manifest_patient_leakage_fails(tmp_path, monkeypatch):
    mod, _, _, manifest, frozen, argv = _anatomy_fixture(tmp_path, monkeypatch)
    _write_manifest(manifest, ['1_10', '1_20', '11050_1001070'])
    with pytest.raises(SystemExit, match='patient overlap'):
        _run(mod, argv + ['--dry-run'])
