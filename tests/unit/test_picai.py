"""PI-CAI 数据审计核心单元测试：标签映射 / 物理角点 / 方向码 / 患者泄漏 / 几何分类 / 分区标签值。"""
import math

import numpy as np
import pytest

from zonal_reliability_fusion.data import picai as P

IDENT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# ---------------- 标签映射 ----------------
def test_map_lesion_resampled_isup_to_foreground():
    arr = np.array([0, 2, 3, 4, 5], dtype=np.uint8)
    out = P.map_lesion_to_binary(arr, P.SRC_RESAMPLED)
    assert out.tolist() == [0, 1, 1, 1, 1]


def test_map_lesion_pooch25_binary():
    arr = np.array([0, 1, 1, 0], dtype=np.uint8)
    out = P.map_lesion_to_binary(arr, P.SRC_POOCH25)
    assert out.tolist() == [0, 1, 1, 0]


def test_map_lesion_resampled_background_zero():
    arr = np.array([0, 0, 0], dtype=np.uint8)
    assert P.map_lesion_to_binary(arr, P.SRC_RESAMPLED).sum() == 0


def test_map_lesion_pooch25_value2_not_foreground():
    # Pooch25 只认值 1；resampled 的值 2 在 Pooch25 语义下不是前景
    arr = np.array([0, 2, 1], dtype=np.uint8)
    assert P.map_lesion_to_binary(arr, P.SRC_POOCH25).tolist() == [0, 0, 1]


# ---------------- 物理角点 / bbox / extent ----------------
def make_geo(size=(2, 3, 4), spacing=(1.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0), direction=IDENT):
    return P.Geometry(size=size, spacing=spacing, origin=origin, direction=direction)


def test_physical_corners_count_and_bbox():
    geo = make_geo()
    corners = P.physical_corners(geo)
    assert corners.shape == (8, 3)
    mn, mx = P.physical_bbox(geo)
    # identity 方向: 物理 = origin + index*spacing；角点 index ∈ {0, size-1}
    assert np.allclose(mn, [10.0, 20.0, 30.0])
    assert np.allclose(mx, [10 + 1 * 1, 20 + 2 * 2, 30 + 3 * 3])  # (11, 24, 39)


def test_physical_extent_not_size_times_spacing():
    geo = make_geo()
    ext = P.physical_extent(geo)
    # 角点 extent = (size-1)*spacing，而非 size*spacing
    assert np.allclose(ext, [1 * 1, 2 * 2, 3 * 3])  # (1,4,9)
    assert not np.allclose(ext, [2 * 1, 3 * 2, 4 * 3])  # 非 size*spacing


def test_orientation_identity_is_LPS():
    # SimpleITK/ITK 使用 DICOM LPS 物理坐标约定：单位 direction → "LPS"（而非 "RAS"）
    assert P.orientation_code(make_geo()) == "LPS"


def test_orientation_single_axis_flip():
    # 单轴翻转（+x 指向 R，其余不变）：LPS 约定下 → "RPS"
    d = (-1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)
    assert P.orientation_code(make_geo(direction=d)) == "RPS"


def test_orientation_oblique_direction_returns_valid_code():
    # 非单位斜扫 direction（绕 z 旋转 20°）：须返回合法三字符方向码
    c, s = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
    d = (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    code = P.orientation_code(make_geo(direction=d))
    assert isinstance(code, str) and len(code) == 3
    # 合法方向码：L/R、P/A、I/S 三对轴字母各恰好出现一次
    assert sum(ch in "LR" for ch in code) == 1
    assert sum(ch in "PA" for ch in code) == 1
    assert sum(ch in "IS" for ch in code) == 1
    # 20° 平面内旋转的主方向仍为 L/P/S
    assert code == "LPS"


# ---------------- 几何分类 ----------------
def test_classify_exact_same_grid():
    a = make_geo()
    b = make_geo()
    assert P.classify_grid(a, b) == P.GRID_EXACT
    assert P.grids_equal(a, b)


def test_classify_same_space_different_grid():
    # 同 origin/direction，spacing 与 size 不同但物理覆盖高度重合(>=0.99) → different
    a = make_geo(size=(200, 200, 20), spacing=(1, 1, 3))
    b = make_geo(size=(100, 100, 20), spacing=(2, 2, 3))  # extent≈(198,198,57) vs (199,199,57)
    st = P.classify_grid(a, b)
    assert st == P.GRID_DIFFERENT


def test_classify_partial_overlap():
    a = make_geo(size=(10, 10, 10), spacing=(1, 1, 1), origin=(0, 0, 0))
    b = make_geo(size=(10, 10, 10), spacing=(1, 1, 1), origin=(5, 5, 5))  # 偏移约一半
    assert P.classify_grid(a, b) == P.GRID_PARTIAL


def test_classify_suspect_no_overlap():
    a = make_geo(size=(10, 10, 10), spacing=(1, 1, 1), origin=(0, 0, 0))
    b = make_geo(size=(10, 10, 10), spacing=(1, 1, 1), origin=(100, 0, 0))  # 完全不相交
    assert P.classify_grid(a, b) == P.GRID_SUSPECT


# ---------------- 患者泄漏 ----------------
def test_find_patient_leaks_detects_overlap():
    splits = {"train": ["p1", "p2", "p3"], "validation": ["p3", "p4"], "test": ["p1"]}
    leaks = P.find_patient_leaks(splits)
    assert set(leaks) == {"p1", "p3"}


def test_find_patient_leaks_clean():
    splits = {"train": ["p1", "p2"], "validation": ["p3"], "test": ["p4"]}
    assert P.find_patient_leaks(splits) == []
    P.validate_disjoint(splits)  # 不抛异常


def test_validate_disjoint_raises():
    with pytest.raises(AssertionError):
        P.validate_disjoint({"train": ["p1"], "validation": ["p1"]})


# ---------------- 分区标签值（真实数据抽样） ----------------
@pytest.mark.skipif(not (P.DEFAULT_DATA_ROOT / P.MARKSHEET_REL).exists(),
                    reason="PI-CAI 数据不可用")
def test_zonal_label_values_are_0_1_2():
    root = P.DEFAULT_DATA_ROOT
    ms = P.load_marksheet(root)
    row = ms[0]
    for sub, known in ((P.ZONAL_YUAN23, {0, 1, 2}), (P.ZONAL_HEVIAI23, {0, 1, 2})):
        p = P.label_path(root, sub, row["patient_id"], row["study_id"])
        assert p.exists(), f"缺少 {sub}"
        uv = set(int(x) for x in np.unique(P.load_array(p)).tolist())
        assert uv.issubset(known), f"{sub} 出现非法标签 {uv - known}"


@pytest.mark.skipif(not (P.DEFAULT_DATA_ROOT / P.MARKSHEET_REL).exists(),
                    reason="PI-CAI 数据不可用")
def test_wg_label_values_are_0_1():
    root = P.DEFAULT_DATA_ROOT
    row = P.load_marksheet(root)[0]
    p = P.label_path(root, P.WG_BOSMA22B, row["patient_id"], row["study_id"])
    uv = set(int(x) for x in np.unique(P.load_array(p)).tolist())
    assert uv.issubset({0, 1}), f"WG 出现非法标签 {uv - {0,1}}"


# ---------------- canonical 来源 ----------------
def test_canonical_prefers_pooch25(tmp_path):
    root = tmp_path
    (root / P.LESION_POOCH25).mkdir(parents=True)
    (root / P.LESION_RESAMPLED).mkdir(parents=True)
    pp = P.label_path(root, P.LESION_POOCH25, 1, 2)
    rr = P.label_path(root, P.LESION_RESAMPLED, 1, 2)
    pp.touch(); rr.touch()
    src, path = P.canonical_lesion(root, 1, 2)
    assert src == P.SRC_POOCH25 and path == pp
