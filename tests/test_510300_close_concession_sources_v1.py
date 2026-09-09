"""验证身份、缺失值、时钟和盘后字段边界，测试不联网。"""

import copy
import json
import unittest
import xml.etree.ElementTree as ET
from decimal import Decimal

from scripts.analyze_510300_close_concession_sources_v1 import (
    DEFAULT_REPORT, SELECT, compare_pcf, iopv_state, number,
    observation_state, pcf_from_xml, read_jsonp, snapshot,
)


class SourceSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = DEFAULT_REPORT / "source_batches"
        cls.xml = (source / "07_pcf_content/manager_pcf_xml.raw").read_bytes()
        cls.basic = (source / "01_initial/sse_pcf_basic.raw").read_bytes()
        cls.components = (source / "07_pcf_content/sse_pcf_components.raw").read_bytes()
        cls.snap_raw = (source / "01_initial/sse_snapshot_existing_fields.raw").read_bytes()

    def test_missing_and_explicit_zero_remain_distinct(self):
        self.assertIsNone(number("-", optional=True))
        self.assertIsNone(number(None, optional=True))
        self.assertEqual(number("0.000"), Decimal("0"))
        self.assertEqual(number("73%"), Decimal("0.73"))
        for invalid in (None, "-", "NaN", "Infinity"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                number(invalid)

    def test_wrong_security_and_date_are_rejected(self):
        for tag, value in [("FundInstrumentID", "510500"), ("TradingDay", "20260904")]:
            root = ET.fromstring(self.xml)
            root.find(tag).text = value
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                pcf_from_xml(ET.tostring(root))

    def test_duplicate_components_cannot_satisfy_record_count(self):
        root = ET.fromstring(self.xml)
        rows = root.find("ComponentList")
        rows.append(copy.deepcopy(rows[0]))
        root.find("RecordNumber").text = "301"
        with self.assertRaises(ValueError):
            pcf_from_xml(ET.tostring(root))

    def test_changed_quantity_is_visible(self):
        root = ET.fromstring(self.xml)
        root.find("./ComponentList/Component/Quantity").text = "1601"
        result, _, rows = compare_pcf(ET.tostring(root), self.basic, self.components)
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual([r["field"] for r in rows if not r["match"]], ["Quantity"])

    def test_missing_cash_cannot_be_replaced_by_zero(self):
        source = read_jsonp(self.components)
        target = next(r for r in source["result"] if r["INSTRUMENT_ID"] == "688981")
        self.assertEqual(target["SUBSTITUTION_CASH_AMOUNT"], "-")
        target["SUBSTITUTION_CASH_AMOUNT"] = "0"
        result, _, rows = compare_pcf(self.xml, self.basic, json.dumps(source).encode("utf-8"))
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual([r["field"] for r in rows if not r["match"]], ["SubstitutionCashAmount"])

    def test_missing_rate_cannot_be_replaced_by_zero(self):
        source = read_jsonp(self.components)
        target = next(r for r in source["result"] if r["INSTRUMENT_ID"] == "000596")
        self.assertEqual(target["REDEMPTION_DISCOUNT_RATE"], "-")
        target["REDEMPTION_DISCOUNT_RATE"] = "0%"
        result, _, _ = compare_pcf(self.xml, self.basic, json.dumps(source).encode("utf-8"))
        self.assertEqual(result["mismatch_count"], 1)

    def test_post_close_zero_during_lunch_is_not_execution_evidence(self):
        data = snapshot(self.snap_raw)
        self.assertEqual(data["fp_volume"], 0)
        self.assertEqual(observation_state(data["source_envelope_time"], "2026-09-07T12:12:29+08:00"),
                         "NOT_OBSERVED_IN_POST_CLOSE_LEGAL_WINDOW")
        self.assertEqual(observation_state(150501, "2026-09-07T15:05:02+08:00"),
                         "WINDOW_ONLY_NOT_A_QUALIFIED_SIGNAL")
        self.assertEqual(observation_state(150501, "2026-09-07T07:05:02+00:00"),
                         "WINDOW_ONLY_NOT_A_QUALIFIED_SIGNAL")
        self.assertEqual(observation_state(153100, "2026-09-07T15:31:01+08:00"),
                         "NOT_OBSERVED_IN_POST_CLOSE_LEGAL_WINDOW")
        with self.assertRaises(ValueError):
            observation_state(151299, "2026-09-07T15:12:01+08:00")
        self.assertNotIn("post_close_pending_buy_qty", data)
        self.assertEqual(iopv_state(0), "NOT_VALID_AS_REFERENCE")
        self.assertEqual(iopv_state(data["iopv"]), "VALUE_PRESENT_TIMESTAMP_NOT_PROVEN")

    def test_snapshot_schema_change_does_not_shift_columns(self):
        with self.assertRaises(ValueError):
            snapshot(self.snap_raw, fields=SELECT[:-1])
        bad = read_jsonp(self.snap_raw)
        bad["code"] = "510500"
        with self.assertRaises(ValueError):
            snapshot(json.dumps(bad, default=str).encode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
