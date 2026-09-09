from __future__ import annotations

from pathlib import Path

import yaml

from research.csi300_pit_fundamental_underreaction_official_facts_v1_2 import (
    PARSER_VERSION,
    extract_metrics_from_page_texts,
    find_core_profit_from_summary_tables,
    locate_statement_sections,
)


def test_multi_page_cash_statement_keeps_real_start_page() -> None:
    pages = [
        "主要会计数据 单位：元 归属于上市公司股东的扣除非经常性损益的净利润 80 70",
        "合并资产负债表 单位：元 项目 流动资产 货币资金 100 90 应收账款 20 18 存货 30 28 资产总计 500 450 负债合计 200 190",
        "合并利润表 单位：元 项目 一、营业收入 300 250 营业成本 180 160 三、营业利润 60 50 归属于母公司股东的净利润 40 35",
        "合并现金流量表 2020年1—3月 编制单位：甲公司 单位：元 项目 一、经营活动产生的现金流量：销售商品、提供劳务收到的现金 350 300",
        "经营活动现金流出小计 305 262 经营活动产生的现金流量净额 45 38 母公司现金流量表",
    ]
    sections = locate_statement_sections(pages)
    assert sections["cash"].page_indices[0] == 3
    assert 4 in sections["cash"].page_indices
    result = extract_metrics_from_page_texts(pages, period_type="Q1")
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert result["parser_version"] == PARSER_VERSION
    assert metrics["OPERATING_CASH_FLOW_YTD"]["metric_value_cny"] == 45.0


def test_narrative_heading_without_statement_unit_is_still_rejected() -> None:
    pages = [
        "经营分析 （b）合并利润表项目 1、营业收入下降 2、营业成本下降",
        "合并资产负债表 单位：元 项目 流动资产 货币资金 100 90",
        "法定代表人：甲 合并利润表",
        "2020年1—9月 编制单位：甲公司 单位：元 项目 营业收入 300 250 营业成本 180 160 营业利润 60 50",
        "法定代表人：甲 合并现金流量表",
        "2020年1—9月 编制单位：甲公司 单位：元 项目 经营活动产生的现金流量 销售商品 350 300 经营活动产生的现金流量净额 45 38",
    ]
    sections = locate_statement_sections(pages)
    assert sections["income"].page_indices[0] == 2


def test_heading_and_table_header_at_previous_page_tail_use_next_page_rows() -> None:
    pages = [
        "主要会计数据 单位：元 归属于上市公司股东的扣除非经常性损益的净利润 80 70",
        "合并资产负债表 单位：元 项目 流动资产 货币资金 100 90",
        "3、合并利润表 单位：元 项目 本期发生额 上期发生额",
        "一、营业总收入 300 250 其中：营业收入 300 250 营业成本 180 160 三、营业利润 60 50 归属于母公司股东的净利润 40 35",
        "4、合并现金流量表 单位：元 项目 本期发生额 上期发生额",
        "一、经营活动产生的现金流量 销售商品 350 300 经营活动产生的现金流量净额 45 38",
    ]
    sections = locate_statement_sections(pages)
    assert sections["income"].page_indices[0] == 2
    assert sections["cash"].page_indices[0] == 4


def test_q3_split_core_label_skips_intervening_repeated_header_and_keeps_ytd_first() -> None:
    pages = [
        "第三季度报告 年初至报告期末（1-9月）单位：元",
        "本报告期末 上年度末 单位：元",
    ]
    rows = [
        (0, 1, 3, ["", "年初至报告期末（1-9月）", "上年同期", "增减"]),
        (0, 1, 8, ["归属于上市公司", "-1,337,573,450.10", "-351,685,766.17", "不适用"]),
        (1, 0, 0, ["", "本报告期末", "上年度末", "增减"]),
        (1, 0, 1, ["股东的扣除非经常性损益的净利润", "", "", ""]),
    ]
    evidence = find_core_profit_from_summary_tables(
        pages,
        rows,
        period_type="Q3",
        fallback_unit="元",
    )
    assert evidence is not None
    assert evidence.metric_value_cny == -1_337_573_450.10
    assert evidence.source_raw_value == "-1,337,573,450.10"
    assert '"lookahead_rows":2' in evidence.source_locator


def test_v1_2_config_and_collector_use_isolated_checkpoint_root() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config/csi300_pit_fundamental_underreaction_official_facts_v1_2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["extraction"]["parser_version"] == PARSER_VERSION
    assert config["artifacts"]["checkpoint_root"].endswith("/checkpoints_v1_2")
    assert (
        config["artifacts"]["checkpoint_root"]
        != config["supersedes"]["legacy_checkpoint_root"]
    )
    from scripts import (
        collect_csi300_pit_fundamental_underreaction_official_facts_v1_2 as collector,
    )

    collector._patch_base_collector()
    assert collector._base.PARSER_VERSION == PARSER_VERSION
    assert collector._base.CONFIG_PATH == collector.CONFIG_PATH
    assert collector._base.process_document is collector.process_document
