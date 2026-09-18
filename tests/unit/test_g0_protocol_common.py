"""G0 协议共享逻辑测试（合成配置/临时文件；不读真实医学数据、不初始化 CUDA）。

覆盖：
- 协议块校验（id/status/sees_model_predictions）；
- 未填字段检测与输出目录隔离（项目内 / 禁止训练产物前缀 / 测试外部目录）；
- 哈希、JSON/CSV 写入与运行元数据。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zonal_reliability_fusion.protocols import common as pc


def write_config(tmp_path: Path, **overrides) -> Path:
    doc = {
        "protocol": {
            "id": "G0-R",
            "version": "draft-0.1",
            "status": "DRAFT",
            "sees_model_predictions": False,
        },
        "threshold_decision": {"status": "INSUFFICIENT_DATA", "chosen_threshold_mm": None},
        "hashes": {},
    }
    for key, value in overrides.items():
        doc[key] = value
    path = tmp_path / "cfg.yaml"
    import yaml

    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return path


def test_load_and_validate_protocol_block(tmp_path: Path):
    path = write_config(tmp_path)
    doc = pc.load_yaml_config(path)
    meta = pc.validate_protocol_block(doc, expected_id="G0-R")
    assert meta["id"] == "G0-R" and meta["status"] == "DRAFT"


def test_protocol_block_rejects_wrong_id_and_bad_status(tmp_path: Path):
    doc = pc.load_yaml_config(write_config(tmp_path))
    with pytest.raises(pc.ProtocolConfigError, match="protocol.id"):
        pc.validate_protocol_block(doc, expected_id="G0-E")
    bad = dict(doc)
    bad["protocol"] = {**doc["protocol"], "status": "PASS"}
    with pytest.raises(pc.ProtocolConfigError, match="protocol.status"):
        pc.validate_protocol_block(bad, expected_id="G0-R")


def test_protocol_block_rejects_model_visibility(tmp_path: Path):
    doc = pc.load_yaml_config(write_config(tmp_path))
    doc["protocol"] = {**doc["protocol"], "sees_model_predictions": True}
    with pytest.raises(pc.ProtocolConfigError, match="sees_model_predictions"):
        pc.validate_protocol_block(doc, expected_id="G0-R")


def test_missing_config_file_and_non_mapping(tmp_path: Path):
    with pytest.raises(pc.ProtocolConfigError, match="不存在"):
        pc.load_yaml_config(tmp_path / "nope.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(pc.ProtocolConfigError, match="顶层必须是映射"):
        pc.load_yaml_config(path)


def test_find_unfilled_fields(tmp_path: Path):
    doc = pc.load_yaml_config(write_config(tmp_path))
    unfilled = pc.find_unfilled_fields(
        doc,
        ["protocol.reviewer", "threshold_decision.status", "threshold_decision.chosen_threshold_mm", "hashes"],
    )
    # INSUFFICIENT_DATA / None / 空映射 均视为“未填”（不伪造默认值）
    assert unfilled == [
        "protocol.reviewer",
        "threshold_decision.status",
        "threshold_decision.chosen_threshold_mm",
        "hashes",
    ]


def test_output_isolation_project_internal_and_forbidden(tmp_path: Path, monkeypatch):
    # 项目内合法目录
    ok_dir = pc.PROJECT_ROOT / "outputs/diagnostics/g0_r/_unit_test_dir"
    assert pc.assert_output_dir_isolated(ok_dir) == ok_dir
    # 禁止落入训练产物 / nnU-Net 工作区
    for bad in (
        pc.PROJECT_ROOT / "outputs/checkpoints/g0_r",
        pc.PROJECT_ROOT / "outputs/nnUNet_results/x",
        pc.PROJECT_ROOT / "workdir/nnUNet_preprocessed/x",
    ):
        with pytest.raises(pc.ProtocolConfigError, match="重叠"):
            pc.assert_output_dir_isolated(bad)
    # 项目外：通过 monkeypatch PROJECT_ROOT 允许测试用临时目录（正式使用仍要求项目内）
    monkeypatch.setattr(pc, "PROJECT_ROOT", tmp_path)
    outside = tmp_path / "run"
    assert pc.assert_output_dir_isolated(outside) == outside


def test_resolve_project_path(tmp_path: Path):
    assert pc.resolve_project_path("outputs/x") == pc.PROJECT_ROOT / "outputs/x"
    absolute = tmp_path / "a"
    assert pc.resolve_project_path(absolute) == absolute


def test_hashes_and_writers(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    digest = pc.sha256_file(f)
    assert len(digest) == 64
    hashes = pc.collect_hashes([f, tmp_path / "missing.txt"])
    assert hashes[str(f)] == digest and hashes[str(tmp_path / "missing.txt")] == "MISSING"
    with pytest.raises(pc.ProtocolConfigError, match="文件不存在"):
        pc.sha256_file(tmp_path / "missing.txt")

    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    csv_path = pc.write_rows_csv(tmp_path / "rows.csv", rows, fieldnames=["a", "b"])
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "a,b"

    json_path = pc.write_json(tmp_path / "x.json", {"k": [1, 2]})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"k": [1, 2]}

    with pytest.raises(pc.ProtocolConfigError, match="至少一个字段名"):
        pc.write_rows_csv(tmp_path / "empty.csv", [], fieldnames=None)


def test_build_run_metadata(tmp_path: Path):
    cfg = write_config(tmp_path)
    meta = pc.build_run_metadata(tool="unit-test", config_path=cfg, extra={"k": "v"})
    assert meta["tool"] == "unit-test"
    assert meta["config_sha256"] == pc.sha256_file(cfg)
    assert meta["extra"] == {"k": "v"}
    assert "torch" not in meta  # 工具不得探测/导入 torch
