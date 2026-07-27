"""Gate5a / Gate5b：窄縫閘的兩種場地語意。

2026-07-27 查到的配置錯誤：`NarrowGapSpec.arena_half_extent` 寫死 5.0，
`--arena_size` 從未傳進去，所以中央牆永遠只長到 y=±4.5 m。

| 場地 | 外牆內緣 | 牆端側口 | D0 實測 |
|---|---|---|---|
| 10 m | ±4.5 | 0 m | direct 0%、撞牆 89.4% |
| 12 m | ±5.5 | 1.0 m | direct 0.42%（幾乎全繞牆端）|
| 14 m | ±6.5 | 2.0 m | direct 100%、撞牆 0% |

12 m 的橫向偏移中位數 5.02 m 正落在牆端 4.5 與外牆 5.5 之間 —— 它鑽的是 1.0 m 側口，
不是中央 1.2 m 縫。考慮 OBB buffer 後中央通道餘裕約 0.4 m、側口僅約 0.2 m，
政策卻選更危險的那條 → 不是路徑最佳化，是**外牆距離造成的 LiDAR 情境捷徑**。

因此裁決把窄縫閘拆成兩個語意不同的閘：

* **Gate5a（sealed）**：牆隨場地延伸到外牆，只留中央窄口 → **純測直穿能力**
* **Gate5b（legacy）**：保留固定牆長不隨場地變 → **專測房間尺度捷徑**

這些測試鎖住兩種模式的幾何，特別是「sealed 在任何場地都零側口」與
「legacy 的側口隨場地線性增加」。
"""

import unittest

from narrow_gap_eval import NarrowGapSpec, spec_for_arena


GAP = 1.2


class SealedModeTest(unittest.TestCase):
    """Gate5a：牆必須封死到外牆，任何場地尺寸皆然。"""

    def test_sealed_leaves_no_side_opening_at_any_arena(self):
        for arena in (10.0, 12.0, 13.0, 14.0, 20.0):
            with self.subTest(arena=arena):
                spec = spec_for_arena(arena, gap_width=GAP, sealed=True)
                self.assertAlmostEqual(spec.side_opening_m(arena), 0.0, places=9)

    def test_sealed_wall_top_equals_boundary_inner_face(self):
        spec = spec_for_arena(14.0, gap_width=GAP, sealed=True)
        wall_top = spec.segment_center_y + 0.5 * spec.segment_length
        self.assertAlmostEqual(wall_top, 14.0 / 2 - 0.5, places=9)

    def test_sealed_keeps_the_central_gap_exactly(self):
        """封死側口不能連帶改變中央缺口寬度。"""
        for arena in (10.0, 14.0):
            spec = spec_for_arena(arena, gap_width=GAP, sealed=True)
            wall_bottom = spec.segment_center_y - 0.5 * spec.segment_length
            self.assertAlmostEqual(2.0 * wall_bottom, GAP, places=9)

    def test_sealed_segment_grows_with_arena(self):
        s10 = spec_for_arena(10.0, gap_width=GAP, sealed=True)
        s14 = spec_for_arena(14.0, gap_width=GAP, sealed=True)
        self.assertAlmostEqual(s14.segment_length - s10.segment_length, 2.0, places=9)


class LegacyModeTest(unittest.TestCase):
    """Gate5b：牆長固定不隨場地變，側口隨場地線性增加。"""

    def test_legacy_side_opening_matches_measured_geometry(self):
        # 對照 2026-07-27 實測表
        for arena, expected in ((10.0, 0.0), (12.0, 1.0), (13.0, 1.5), (14.0, 2.0)):
            with self.subTest(arena=arena):
                spec = spec_for_arena(arena, gap_width=GAP, sealed=False)
                self.assertAlmostEqual(spec.side_opening_m(arena), expected, places=9)

    def test_legacy_wall_geometry_is_independent_of_arena(self):
        s10 = spec_for_arena(10.0, gap_width=GAP, sealed=False)
        s14 = spec_for_arena(14.0, gap_width=GAP, sealed=False)
        self.assertAlmostEqual(s10.segment_length, s14.segment_length, places=9)
        self.assertAlmostEqual(s10.segment_center_y, s14.segment_center_y, places=9)

    def test_legacy_reproduces_the_historical_hardcoded_spec(self):
        """歷史上所有 Gate5 都等價於 legacy@10m，不得因本次改動而改變。"""
        legacy = spec_for_arena(10.0, gap_width=GAP, sealed=False)
        historical = NarrowGapSpec(gap_width=GAP)   # arena_half_extent 預設 5.0
        self.assertAlmostEqual(legacy.segment_length, historical.segment_length, places=9)
        self.assertAlmostEqual(
            legacy.segment_center_y, historical.segment_center_y, places=9
        )


class ModesAgreeAtTenMetresTest(unittest.TestCase):
    def test_two_modes_are_identical_at_the_historical_size(self):
        """10 m 是唯一兩模式重合的尺寸 —— 這正是錯配未被發現的原因。"""
        sealed = spec_for_arena(10.0, gap_width=GAP, sealed=True)
        legacy = spec_for_arena(10.0, gap_width=GAP, sealed=False)
        self.assertAlmostEqual(sealed.segment_length, legacy.segment_length, places=9)
        self.assertAlmostEqual(sealed.segment_center_y, legacy.segment_center_y, places=9)

    def test_modes_diverge_above_ten_metres(self):
        sealed = spec_for_arena(14.0, gap_width=GAP, sealed=True)
        legacy = spec_for_arena(14.0, gap_width=GAP, sealed=False)
        self.assertGreater(sealed.segment_length, legacy.segment_length)


class ValidationTest(unittest.TestCase):
    def test_sealed_requires_an_explicit_arena_size(self):
        """少了場地尺寸就會退回寫死的 5.0 —— 那正是 07-27 錯配的成因。"""
        with self.assertRaises(ValueError):
            spec_for_arena(None, gap_width=GAP, sealed=True)

    def test_legacy_tolerates_a_missing_arena_size(self):
        """legacy 本來就固定按 10 m 算，缺場地尺寸不改變它的語意。"""
        implied = spec_for_arena(None, gap_width=GAP, sealed=False)
        explicit = spec_for_arena(10.0, gap_width=GAP, sealed=False)
        self.assertAlmostEqual(
            implied.segment_length, explicit.segment_length, places=9
        )

    def test_rejects_nonpositive_arena(self):
        with self.assertRaises(ValueError):
            spec_for_arena(0.0, gap_width=GAP, sealed=True)

    def test_rejects_gap_wider_than_arena(self):
        with self.assertRaises(ValueError):
            spec_for_arena(1.0, gap_width=GAP, sealed=True)


if __name__ == "__main__":
    unittest.main()
