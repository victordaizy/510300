from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import json
import re
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser


DIRECT_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_DIRECT_PDF_DOCUMENT_BY_DOCUMENT_STATISTICS_V1_3_40"
)
DIRECT_VERIFICATION_STATUS = (
    "PASS_OFFICIAL_ORIGINAL_PDF_DIRECT_ACCOUNTING_IDENTITY_RECONCILED"
)
BALANCE_TARGET_METRICS = {
    "ACCOUNTS_RECEIVABLE_END",
    "INVENTORY_END",
    "TOTAL_ASSETS_END",
    "TOTAL_LIABILITIES_END",
}
INCOME_TARGET_METRICS = {
    "OPERATING_REVENUE_YTD",
    "OPERATING_PROFIT_YTD",
    "PARENT_NET_PROFIT_YTD",
}

MANUAL_DIRECT_TEXT_ROW_FACTS: dict[
    str,
    dict[str, Any] | tuple[dict[str, Any], ...],
] = {
    "1202268049": {
        "metric_id": "PARENT_NET_PROFIT_YTD",
        "page_number": 4,
        "source_section_label": "2.1.2按中国企业会计准则编制的主要会计数据及财务指标",
        "label": "归属于母公司股东的净（亏损）/利润",
        "unit": "百万元",
        "current": "-13786",
        "prior": "6149",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1202636649": {
        "metric_id": "TOTAL_LIABILITIES_END",
        "page_number": 28,
        "label": "负债合计",
        "label_pattern": r"(?<!流动)(?<!非流动)负债合计",
        "unit": "元",
        "current": "5526466486.70",
        "prior": "4072203945.04",
        "value_period_scope": "PERIOD_END",
    },
    "1203217917": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 78,
            "label": "应收账款",
            "unit": "元",
            "current": "83436913111",
            "prior": "63845777577",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 78,
            "label": "存货",
            "unit": "元",
            "current": "131527028973",
            "prior": "125742666785",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 79,
            "label": "负债合计",
            "label_pattern": r"(?<!流动)(?<!非流动)负债合计",
            "unit": "元",
            "current": "614505763527",
            "prior": "561487988536",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1204557748": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 94,
            "label": "应收账款",
            "unit": "元",
            "current": "68044732239",
            "prior": "83436913111",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 94,
            "label": "存货",
            "unit": "元",
            "current": "130113267841",
            "prior": "131527028973",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 95,
            "label": "负债合计",
            "label_pattern": r"(?<!流动)(?<!非流动)负债合计",
            "unit": "元",
            "current": "644293522111",
            "prior": "614505763527",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1204682861": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 25,
            "source_section_label": "1、资产构成重大变动情况",
            "label": "应收账款",
            "unit": "元",
            "current": "1224109901.37",
            "prior": "663524805.12",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 25,
            "source_section_label": "1、资产构成重大变动情况",
            "label": "存货",
            "unit": "元",
            "current": "1923157726.14",
            "prior": "1949548273.48",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1205363576": {
        "metric_id": "ACCOUNTS_RECEIVABLE_END",
        "page_number": 135,
        "label": "应收账款",
        "label_pattern": r"应收账款(?=\(b\)125,472,252)",
        "unit": "千元",
        "current": "125472252",
        "prior": "111127157",
        "value_period_scope": "PERIOD_END",
    },
    "1205947423": {
        "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
        "page_number": 227,
        "label": "扣除非经常性损益后归属于本公司普通股股东的净利润",
        "unit": "千元",
        "current": "6620005",
        "prior": "7226790",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1205969212": {
        "metric_id": "ACCOUNTS_RECEIVABLE_END",
        "page_number": 205,
        "label": "应收账款",
        "label_pattern": r"应收账款(?=\(b\)105,909,473)",
        "unit": "千元",
        "current": "105909473",
        "prior": "111127157",
        "value_period_scope": "PERIOD_END",
    },
    "1206869181": (
        {
            "metric_id": "INVENTORY_END",
            "page_number": 26,
            "source_section_label": "1.资产及负债状况",
            "label": "存货",
            "unit": "千元",
            "current": "214935934",
            "prior": "165241259",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 129,
            "source_section_label": "(5) 应收账款",
            "label": "应收账款",
            "label_pattern": r"应收账款(?=118,940,688)",
            "unit": "千元",
            "current": "118940688",
            "prior": "110796700",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1207691674": (
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 18,
            "label": "营业（亏损）/利润",
            "unit": "百万元",
            "current": "-9252",
            "prior": "29270",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 18,
            "label": "归属于母公司股东的净（亏损）/利润",
            "unit": "百万元",
            "current": "-16234",
            "prior": "10245",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1208326870": (
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 9,
            "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
            "unit": "千元",
            "current": "-9596763",
            "prior": "3025432",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 55,
            "label": "营业(亏损)利润",
            "unit": "千元",
            "current": "-13086430",
            "prior": "4382127",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1208321741": (
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 65,
            "label": "二、营业(亏损)/利润",
            "unit": "百万元",
            "current": "-12300",
            "prior": "2079",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 65,
            "label": "归属于母公司股东的净(亏损)/利润",
            "unit": "百万元",
            "current": "-8174",
            "prior": "1690",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1208345210": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "source_section_label": "1 按中国企业会计准则编制的财务数据和指标",
            "label": "归属于母公司股东的净（亏损）/利润",
            "unit": "百万元",
            "current": "-22882",
            "prior": "31338",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "source_section_label": "1 按中国企业会计准则编制的财务数据和指标",
            "label": "归属于母公司股东的扣除非经常性损益后的净（亏损）/利润",
            "unit": "百万元",
            "current": "-24404",
            "prior": "30451",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 28,
            "label": "合并营业（亏损）/利润",
            "unit": "百万元",
            "current": "-27723",
            "prior": "49178",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1208637839": {
        "metric_id": "TOTAL_LIABILITIES_END",
        "page_number": 28,
        "label": "负债合计",
        "label_pattern": r"(?<!流动)(?<!非流动)负债合计",
        "unit": "千元",
        "current": "120511299",
        "prior": "103247837",
        "value_period_scope": "PERIOD_END",
        "value_search_scope": "PAGE",
    },
    "1209489664": (
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 102,
            "label": "二、营业(亏损)/利润",
            "unit": "百万元",
            "current": "-15641",
            "prior": "3202",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 102,
            "label": "归属于母公司股东的净(亏损)/利润",
            "unit": "百万元",
            "current": "-10842",
            "prior": "2651",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1210899453": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-4688",
            "prior": "-8174",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-4800",
            "prior": "-8418",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1212748163": (
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 120,
            "label": "二、营业亏损",
            "unit": "百万元",
            "current": "-14302",
            "prior": "-15641",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的净（亏损）/利润",
            "unit": "百万元",
            "current": "-12103",
            "prior": "-10842",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1214471034": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-11488",
            "prior": "-4688",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-11751",
            "prior": "-4800",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1216244194": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-32682",
            "prior": "-12103",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-34028",
            "prior": "-12630",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1217700396": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-2875",
            "prior": "-11488",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-3960",
            "prior": "-11751",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1219428821": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-4209",
            "prior": "-32682",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-6420",
            "prior": "-34028",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1221058205": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-1228",
            "prior": "-2875",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-3464",
            "prior": "-3960",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1222912795": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-1696",
            "prior": "-4209",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-3948",
            "prior": "-6420",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1224612089": (
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的净亏损",
            "unit": "百万元",
            "current": "-1533",
            "prior": "-1228",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 8,
            "label": "归属于上市公司股东的扣除非经常性损益的净亏损",
            "unit": "百万元",
            "current": "-2033",
            "prior": "-3464",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1214469292": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 84,
            "unit_page_number": 49,
            "label": "应收账款净额",
            "label_pattern": r"合计(?=11,400,828,869\.24)",
            "unit": "元",
            "current": "11400828869.24",
            "prior": "7986787322.58",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 94,
            "unit_page_number": 49,
            "label": "存货净额",
            "label_pattern": r"合计(?=18,852,196,350\.01)",
            "unit": "元",
            "current": "17424490828.71",
            "prior": "19062432842.88",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 38,
            "label": "负债合计",
            "label_pattern": r"(?<!流动)(?<!非流动)负债合计",
            "unit": "元",
            "current": "73248089447.72",
            "prior": "67720696078.96",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1210899621": {
        "metric_id": "TOTAL_LIABILITIES_END",
        "page_number": 51,
        "label": "负债总计",
        "unit": "元",
        "current": "93220641153",
        "prior": "74022247962",
        "value_period_scope": "PERIOD_END",
    },
    "1214474223": (
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
            "unit": "千元",
            "current": "-19494182",
            "prior": "-6920393",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 15,
            "source_section_label": "1、资产及负债状况",
            "label": "应收账款",
            "unit": "千元",
            "current": "1638492",
            "prior": "2991037",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1213176408": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 5,
            "unit_page_number": 4,
            "label": "应收账款",
            "unit": "元",
            "current": "23484844663.54",
            "prior": "18970494098.56",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 5,
            "unit_page_number": 4,
            "label": "存货",
            "unit": "元",
            "current": "157030124.11",
            "prior": "130963317.78",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 6,
            "unit_page_number": 4,
            "label": "负债合计",
            "label_pattern": r"(?<!流动)(?<!非流动)负债合计",
            "unit": "元",
            "current": "160427011853.66",
            "prior": "140601171304.61",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1217717273": {
        "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
        "page_number": 7,
        "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
        "unit": "千元",
        "current": "-4938159",
        "prior": "-19494182",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1221057925": {
        "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
        "page_number": 7,
        "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
        "unit": "千元",
        "current": "-3440209",
        "prior": "-4938159",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1224611848": {
        "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
        "page_number": 7,
        "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
        "unit": "千元",
        "current": "-2003580",
        "prior": "-3440209",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1224755022": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 6,
            "label": "应收账款",
            "unit": "元",
            "current": "6939356414.14",
            "prior": "9196758436.92",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 6,
            "label": "存货",
            "unit": "元",
            "current": "20822456618.83",
            "prior": "20303476342.94",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 8,
            "label": "负债合计",
            "label_pattern": r"(?<!流动)(?<!非流动)负债合计",
            "unit": "元",
            "current": "271982816471.06",
            "prior": "253147278222.88",
            "value_period_scope": "PERIOD_END",
        },
    ),
}

MANUAL_DIRECT_TEXT_RECEIVABLE_RECONCILIATION_FACTS: dict[str, dict[str, Any]] = {
    "1205947423": {
        "page_number": 138,
        "unit": "千元",
        "current_gross": "5590112",
        "current_allowance": "216140",
        "current_bills": "403",
        "current_combined": "5374375",
        "prior_gross": "3674827",
        "prior_allowance": "184400",
        "prior_bills": "348",
        "prior_combined": "3490775",
    },
}

ASSET_PATTERNS = (r"(?<!流动)(?<!非流动)资产总计",)
LIABILITY_PATTERNS = (r"(?<!流动)(?<!非流动)负债(?:合计|总计)",)
EQUITY_PATTERNS = (
    r"(?<!公司)(?<!母公司)(?<!少数)"
    r"(?:股东权益|所有者权益)(?:（或股东权益）)?合计",
)
LIABILITY_EQUITY_TOTAL_PATTERNS = (
    r"负债(?:及|和)(?:股东权益|所有者权益)"
    r"(?:（或股东权益）)?(?:总计|合计)",
)
RECEIVABLE_PATTERNS = (
    r"(?<!应收票据及)(?<!应收票据和)(?<!其他)"
    r"应收账款(?!及应收票据)(?!和应收票据)(?!及合同资产)(?!和合同资产)",
)
INVENTORY_PATTERNS = (r"(?<!周转)存货(?!跌价)(?!减值)",)
REVENUE_PATTERNS = (
    r"(?:[一二三四五六七八九十][、.]?)?营业收入",
    r"(?:[一二三四五六七八九十][、.]?)?营业总收入",
)
OPERATING_PROFIT_PATTERNS = (
    r"(?:[一二三四五六七八九十][、.]?)?营业利润"
    r"(?:\s*[（(]亏损[^）)]{0,40}[）)])?",
)
NET_PROFIT_PATTERNS = (
    r"(?:[一二三四五六七八九十][、.]?)?净利润",
    r"(?<!持续经营)(?<!终止经营)(?<!母公司)(?<!的)净利润"
    r"(?:\s*[（(]净亏损[^）)]{0,40}[）)])?",
)
PARENT_PROFIT_PATTERNS = (
    r"归属于母公司(?:普通股)?(?:所有者|股东)(?:的)?净利润",
    r"归属于本公司股东的净利润",
)
MINORITY_PROFIT_PATTERNS = (r"少数股东(?:损益|损失)",)

_SUMMARY_UNIT_PATTERN = r"(?P<unit>元|千元|万元|百万元|亿元)"
SUMMARY_METRIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "OPERATING_REVENUE_YTD": (
        rf"(?<!总)营业收入\s*[（(](?:人民币)?{_SUMMARY_UNIT_PATTERN}[）)]",
    ),
    "PARENT_NET_PROFIT_YTD": (
        rf"归属于(?:母公司|本公司|上市公司)(?:普通股)?(?:所有者|股东)的净利润"
        rf"\s*[（(](?:人民币)?{_SUMMARY_UNIT_PATTERN}[）)]",
    ),
    "CORE_PARENT_NET_PROFIT_YTD": (
        rf"归属于(?:母公司|本公司|上市公司)(?:普通股)?(?:所有者|股东)(?:的)?"
        rf"扣除非经常性损益(?:后)?的净利润\s*[（(](?:人民币)?"
        rf"{_SUMMARY_UNIT_PATTERN}[）)]",
    ),
    "OPERATING_CASH_FLOW_YTD": (
        rf"经营活动产生的现金流量净额\s*[（(](?:人民币)?"
        rf"{_SUMMARY_UNIT_PATTERN}[）)]",
    ),
    "TOTAL_ASSETS_END": (
        rf"(?<!净)总资产\s*[（(](?:人民币)?{_SUMMARY_UNIT_PATTERN}[）)]",
    ),
}


def _metric_cny(result: dict[str, Any], metric_id: str) -> Decimal | None:
    return parser._v1_6._metric_cny(result, metric_id)


def _metric_source_unit(result: dict[str, Any], metric_id: str) -> str | None:
    for metric in result.get("metrics") or []:
        if str(metric.get("metric_id")) != metric_id:
            continue
        unit = str(metric.get("source_unit") or "")
        return unit if unit in parser._base.UNIT_MULTIPLIERS else None
    return None


def _pages(start: int, end: int, count: int) -> tuple[int, ...]:
    return tuple(range(max(0, start), min(count, end)))


def _pair(
    page_texts: Sequence[str],
    pages: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    return parser._substantial_amount_pair(
        page_texts,
        pages,
        patterns,
        unit=unit,
    )


def _whole_page_pair(
    page_texts: Sequence[str],
    pages: Iterable[int],
    patterns: Sequence[str],
    *,
    unit: str,
    amount_index: int,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    for page_index in pages:
        pair = _statement_page_pair(
            page_texts[page_index],
            page_number=page_index + 1,
            patterns=patterns,
            unit=unit,
            amount_index=amount_index,
        )
        if pair is not None:
            return pair
    return None


def _statement_page_pair(
    page_text: str,
    *,
    page_number: int,
    patterns: Sequence[str],
    unit: str,
    amount_index: int,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    """读取报表金额列，并排除金额列前的附注编号。"""

    for pattern_text in patterns:
        label_match = re.search(pattern_text, page_text)
        if label_match is None:
            continue
        amount_matches: list[tuple[re.Match[str], str, Decimal]] = []
        tail = page_text[: min(len(page_text), label_match.end() + 700)]
        for amount_match in parser._base._number_matches_after(tail, label_match.end()):
            raw_value = parser._raw_amount_with_closing_parenthesis(
                page_text,
                amount_match,
            )
            if re.search(r"\d", raw_value) is None:
                continue
            value = parser._base.parse_decimal(raw_value)
            if value == value.to_integral_value() and abs(value) <= 999:
                continue
            if value == value.to_integral_value() and 1900 <= value <= 2100:
                continue
            amount_matches.append((amount_match, raw_value, value))
        if amount_index + 1 >= len(amount_matches):
            continue
        selected: list[parser._base.ParsedCandidate] = []
        for selected_index in (amount_index, amount_index + 1):
            amount_match, raw_value, value = amount_matches[selected_index]
            selected.append(
                parser._base.ParsedCandidate(
                    value_cny=value * parser._base.UNIT_MULTIPLIERS[unit],
                    page_number=page_number,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=unit,
                    line_window=page_text[
                        max(0, label_match.start() - 240) : min(
                            len(page_text),
                            amount_match.end() + 240,
                        )
                    ],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=selected_index,
                )
            )
        return selected[0], selected[1]
    return None


def _pair_values(
    pair: tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate],
) -> tuple[Decimal, Decimal]:
    return pair[0].value_cny, pair[1].value_cny


def _same_display(left: Decimal, right: Decimal, unit: str) -> bool:
    return parser._v1_4._same_display_identity(left, right, unit)


def _unique_explicit_unit(
    page_texts: Sequence[str],
    pages: Sequence[int],
) -> str | None:
    unit = parser._base.detect_unique_unit(page_texts, pages)
    return unit if unit in parser._base.UNIT_MULTIPLIERS else None


def _nearest_balance_scope(
    compact_pages: Sequence[str],
    page_index: int,
) -> str | None:
    context = "\n".join(
        compact_pages[index]
        for index in range(max(0, page_index - 4), page_index + 1)
    )
    consolidated_position = max(
        context.rfind("合并资产负债表"),
        context.rfind("合并及公司资产负债表"),
    )
    parent_position = max(
        context.rfind("母公司资产负债表"),
        context.rfind("公司资产负债表"),
    )
    if consolidated_position >= 0 and consolidated_position > parent_position:
        return "EXPLICIT_CONSOLIDATED_BALANCE_HEADING"
    return None


def _direct_evidence(
    metric_id: str,
    candidate: parser._base.ParsedCandidate,
    *,
    value_period_scope: str,
    section_name: str,
    validation: dict[str, Any],
    source_context: dict[str, Any],
) -> parser._base.MetricEvidence:
    locator = {
        "page": candidate.page_number,
        "section": section_name,
        "line_window": candidate.line_window[:700],
        "label_span": list(candidate.match_span),
        "value_span": list(candidate.value_span),
        "selected_amount_index": candidate.selected_amount_index,
        "validation": validation,
        "source_context": source_context,
        "parser_version": DIRECT_PARSER_VERSION,
        "text_engine": "PDFIUM_DIRECT_DOCUMENT_STATISTICS",
    }
    return parser._base.MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(candidate.value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=value_period_scope,
        source_page=candidate.page_number,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=candidate.label,
        source_raw_value=candidate.raw_value,
        source_unit=candidate.unit,
        source_unit_multiplier=float(parser._base.UNIT_MULTIPLIERS[candidate.unit]),
        source_method=(
            "PDFIUM_DIRECT_DOCUMENT_BY_DOCUMENT_DUAL_PERIOD_"
            "ACCOUNTING_IDENTITY_RECONCILED"
        ),
        verification_status=DIRECT_VERIFICATION_STATUS,
    )


def _summary_metric_pair(
    page_texts: Sequence[str],
    metric_id: str,
    *,
    result: dict[str, Any] | None = None,
    period_type: str | None = None,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    candidates: dict[Decimal, tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate]] = {}
    summary_page_texts = tuple(
        _normalize_direct_summary_numeric_layout(page_text)
        for page_text in page_texts[:30]
    )
    for page_index, page_text in enumerate(summary_page_texts):
        for pattern in SUMMARY_METRIC_PATTERNS.get(metric_id, ()):
            label_match = re.search(pattern, page_text)
            if label_match is None:
                continue
            unit = str(label_match.group("unit"))
            if unit not in parser._base.UNIT_MULTIPLIERS:
                continue
            amount_index = _summary_amount_index(
                page_text,
                result=result,
                period_type=period_type,
            )
            if amount_index is None:
                continue
            pair = _statement_page_pair(
                page_text,
                page_number=page_index + 1,
                patterns=(pattern,),
                unit=unit,
                amount_index=amount_index,
            )
            if pair is None:
                continue
            label_to_first_amount = page_text[
                label_match.end() : pair[0].value_span[0]
            ]
            if re.search(r"[\u4e00-\u9fff]", label_to_first_amount):
                continue
            candidates[pair[0].value_cny] = pair
    if metric_id == "CORE_PARENT_NET_PROFIT_YTD":
        for pair in _cross_page_core_summary_pairs(
            summary_page_texts,
            result=result,
            period_type=period_type,
        ):
            candidates[pair[0].value_cny] = pair
        for pair in _table_unit_core_summary_pairs(
            summary_page_texts,
            result=result,
            period_type=period_type,
        ):
            candidates[pair[0].value_cny] = pair
    if metric_id == "PARENT_NET_PROFIT_YTD":
        for pair in _table_unit_parent_summary_pairs(
            summary_page_texts,
            period_type=period_type,
        ):
            candidates[pair[0].value_cny] = pair
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _cross_page_core_summary_pairs(
    page_texts: Sequence[str],
    *,
    result: dict[str, Any] | None,
    period_type: str | None,
) -> list[tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate]]:
    """处理季度报告首页表格把扣非归母标签拆到相邻两页的情形。"""

    split_patterns = (
        (
            r"归属于(?:母公司|本公司|上市公司)(?:普通股)?"
            r"(?:所有者|股东)(?:的)?扣除",
            rf"非经常性损益(?:后)?的净利润\s*[（(](?:人民币)?"
            rf"{_SUMMARY_UNIT_PATTERN}[）)]",
        ),
        (
            r"归属于(?:母公司|本公司|上市公司)(?:普通股)?"
            r"(?:所有者|股东)(?:的)?扣除非经常性损益(?:后)?",
            rf"的净利润\s*[（(](?:人民币)?{_SUMMARY_UNIT_PATTERN}[）)]",
        ),
    )
    pairs: list[
        tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate]
    ] = []
    for page_index in range(min(len(page_texts) - 1, 29)):
        page_text = page_texts[page_index]
        next_page_text = page_texts[page_index + 1]
        for prefix_pattern, continuation_pattern in split_patterns:
            prefix_match = re.search(prefix_pattern, page_text)
            continuation_match = re.search(continuation_pattern, next_page_text[:800])
            if prefix_match is None or continuation_match is None:
                continue
            unit = str(continuation_match.group("unit"))
            amount_index = _summary_amount_index(
                page_text,
                result=result,
                period_type=period_type,
            )
            if amount_index is None:
                continue
            pair = _statement_page_pair(
                page_text,
                page_number=page_index + 1,
                patterns=(prefix_pattern,),
                unit=unit,
                amount_index=amount_index,
            )
            if pair is None:
                continue
            label_to_first_amount = page_text[
                prefix_match.end() : pair[0].value_span[0]
            ]
            if re.search(r"[\u4e00-\u9fff]", label_to_first_amount):
                continue
            combined_label = prefix_match.group(0) + continuation_match.group(0)
            continuation_excerpt = next_page_text[
                max(0, continuation_match.start() - 80) : continuation_match.end() + 80
            ]
            rebuilt: list[parser._base.ParsedCandidate] = []
            for candidate in pair:
                rebuilt.append(
                    parser._base.ParsedCandidate(
                        value_cny=candidate.value_cny,
                        page_number=candidate.page_number,
                        label=combined_label,
                        raw_value=candidate.raw_value,
                        unit=candidate.unit,
                        line_window=(
                            candidate.line_window
                            + f"\n[相邻页标签续接：PDF第{page_index + 2}页]"
                            + continuation_excerpt
                        ),
                        match_span=candidate.match_span,
                        value_span=candidate.value_span,
                        selected_amount_index=candidate.selected_amount_index,
                    )
                )
            pairs.append((rebuilt[0], rebuilt[1]))
    return pairs


def _normalize_direct_summary_numeric_layout(value: str) -> str:
    normalized = parser._normalize_summary_numeric_layout(value)
    for _ in range(3):
        previous = normalized
        normalized = re.sub(
            r"(?<=,)\s+(?=\d{3}(?:\.\d+)?(?:\D|$))",
            "",
            normalized,
        )
        normalized = parser._normalize_summary_numeric_layout(normalized)
        if normalized == previous:
            break
    return normalized


def _table_unit_core_summary_pairs(
    page_texts: Sequence[str],
    *,
    result: dict[str, Any] | None,
    period_type: str | None,
) -> list[tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate]]:
    """读取单位位于表头、而非逐行重复的中国准则 Q3 扣非归母数据。"""

    if period_type != "Q3":
        return []
    _ = result
    core_patterns = (
        r"归属于(?:母公司|本公司|上市公司)(?:普通股)?"
        r"(?:所有者|股东)(?:的)?扣除非经常性损益(?:后)?的净利润"
        r"(?:\s*/?\s*[（(]亏损[）)])?",
    )
    pairs: list[
        tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate]
    ] = []
    for page_index, page_text in enumerate(page_texts):
        if re.search(core_patterns[0], page_text) is None:
            continue
        compact = parser._v1_4.compact_financial_text(page_text)
        if "3个月" not in compact or "9个月" not in compact:
            continue
        scope_context = "\n".join(page_texts[max(0, page_index - 1) : page_index + 1])
        if "中国企业会计准则" not in scope_context:
            continue
        unit = _unique_explicit_unit(page_texts, (page_index,))
        if unit is None:
            continue
        pair = _q3_six_column_table_pair(
            page_text,
            page_number=page_index + 1,
            patterns=core_patterns,
            unit=unit,
        )
        if pair is not None:
            pairs.append(pair)
    return pairs


def _q3_six_column_table_pair(
    page_text: str,
    *,
    page_number: int,
    patterns: Sequence[str],
    unit: str,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    """从“单季三列 + 年初至今三列”的表格行读取第 4、5 列。"""

    cell_pattern = re.compile(
        r"(?<![\d,])(?:\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?\)?|(?<!\S)[-—–](?!\S))(?![\d,])"
    )
    for pattern_text in patterns:
        label_match = re.search(pattern_text, page_text)
        if label_match is None:
            continue
        region_end = min(len(page_text), label_match.end() + 500)
        for marker in (
            "加权平均净资产收益率",
            "基本每股收益",
            "稀释每股收益",
            "经营活动产生的现金流量净额",
        ):
            marker_position = page_text.find(marker, label_match.end())
            if marker_position >= 0:
                region_end = min(region_end, marker_position)
        cell_matches = list(cell_pattern.finditer(page_text, label_match.end(), region_end))
        if len(cell_matches) != 6:
            continue
        selected: list[parser._base.ParsedCandidate] = []
        for selected_index in (3, 4):
            amount_match = cell_matches[selected_index]
            raw_value = amount_match.group(0)
            if raw_value in {"-", "—", "–"}:
                return None
            value = parser._base.parse_decimal(raw_value)
            selected.append(
                parser._base.ParsedCandidate(
                    value_cny=value * parser._base.UNIT_MULTIPLIERS[unit],
                    page_number=page_number,
                    label=label_match.group(0),
                    raw_value=raw_value,
                    unit=unit,
                    line_window=page_text[
                        max(0, label_match.start() - 320) : min(
                            len(page_text), region_end + 120
                        )
                    ],
                    match_span=(label_match.start(), label_match.end()),
                    value_span=(amount_match.start(), amount_match.end()),
                    selected_amount_index=selected_index,
                )
            )
        return selected[0], selected[1]
    return None


def _table_unit_parent_summary_pairs(
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> list[tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate]]:
    """读取单位在列头的中国准则半年报归母净利润。"""

    if period_type != "H1":
        return []
    pairs: list[
        tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate]
    ] = []
    for page_index, page_text in enumerate(page_texts):
        if "中国企业会计准则" not in page_text:
            continue
        compact = parser._v1_4.compact_financial_text(page_text)
        if "6个月期间" not in compact:
            continue
        for pattern in PARENT_PROFIT_PATTERNS:
            label_match = re.search(pattern, page_text)
            if label_match is None:
                continue
            unit = _table_unit_before_label(page_text, label_match.start())
            if unit is None:
                continue
            pair = _statement_page_pair(
                page_text,
                page_number=page_index + 1,
                patterns=(pattern,),
                unit=unit,
                amount_index=0,
            )
            if pair is None:
                continue
            rebuilt: list[parser._base.ParsedCandidate] = []
            for candidate in pair:
                rebuilt.append(
                    parser._base.ParsedCandidate(
                        value_cny=candidate.value_cny,
                        page_number=candidate.page_number,
                        label=candidate.label,
                        raw_value=candidate.raw_value,
                        unit=candidate.unit,
                        line_window=page_text[
                            max(0, label_match.start() - 1_200) : min(
                                len(page_text), candidate.value_span[1] + 240
                            )
                        ],
                        match_span=candidate.match_span,
                        value_span=candidate.value_span,
                        selected_amount_index=candidate.selected_amount_index,
                    )
                )
            pairs.append((rebuilt[0], rebuilt[1]))
    return pairs


def _table_unit_before_label(page_text: str, label_start: int) -> str | None:
    context_start = max(0, label_start - 1_200)
    context = page_text[context_start:label_start]
    units = {
        str(match.group("unit"))
        for match in re.finditer(
            rf"人民币\s*{_SUMMARY_UNIT_PATTERN}",
            context,
        )
        if str(match.group("unit")) in parser._base.UNIT_MULTIPLIERS
    }
    return next(iter(units)) if len(units) == 1 else None


def _summary_amount_index(
    page_text: str,
    *,
    result: dict[str, Any] | None,
    period_type: str | None,
) -> int | None:
    if period_type != "Q3":
        return 0
    if result is None:
        return None
    matched_indices: set[int] = set()
    for anchor_metric_id in (
        "PARENT_NET_PROFIT_YTD",
        "OPERATING_REVENUE_YTD",
        "OPERATING_CASH_FLOW_YTD",
    ):
        expected = _metric_cny(result, anchor_metric_id)
        if expected is None:
            continue
        expected_unit = _metric_source_unit(result, anchor_metric_id)
        for pattern in SUMMARY_METRIC_PATTERNS.get(anchor_metric_id, ()):
            label_match = re.search(pattern, page_text)
            if label_match is None:
                continue
            unit = str(label_match.group("unit"))
            for amount_index in range(9):
                pair = _statement_page_pair(
                    page_text,
                    page_number=1,
                    patterns=(pattern,),
                    unit=unit,
                    amount_index=amount_index,
                )
                if (
                    pair is not None
                    and amount_index in (2, 3)
                    and _same_display(
                    pair[0].value_cny,
                    expected,
                    expected_unit or unit,
                    )
                ):
                    matched_indices.add(amount_index)
    return next(iter(matched_indices)) if len(matched_indices) == 1 else None


def _summary_evidence(
    metric_id: str,
    candidate: parser._base.ParsedCandidate,
    *,
    source_context: dict[str, Any],
) -> parser._base.MetricEvidence:
    value_period_scope = "PERIOD_END" if metric_id.endswith("_END") else "YEAR_TO_DATE"
    locator = {
        "page": candidate.page_number,
        "section": "OFFICIAL_REPORT_EXPLICIT_UNIT_FINANCIAL_SUMMARY",
        "line_window": candidate.line_window[:700],
        "selected_amount_index": candidate.selected_amount_index,
        "validation": {
            "explicit_source_unit": candidate.unit,
            "dual_period_values_present": True,
            "unique_current_value_in_first_30_pages": True,
        },
        "source_context": source_context,
        "parser_version": DIRECT_PARSER_VERSION,
        "text_engine": "PDFIUM_DIRECT_EXPLICIT_UNIT_SUMMARY_STATISTICS",
    }
    return parser._base.MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(candidate.value_cny),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=value_period_scope,
        source_page=candidate.page_number,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=candidate.label,
        source_raw_value=candidate.raw_value,
        source_unit=candidate.unit,
        source_unit_multiplier=float(parser._base.UNIT_MULTIPLIERS[candidate.unit]),
        source_method="PDFIUM_DIRECT_EXPLICIT_UNIT_DUAL_PERIOD_SUMMARY_STATISTICS",
        verification_status=DIRECT_VERIFICATION_STATUS,
    )


def _add_direct_summary_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    admitted: list[str] = []
    source_pages: set[int] = set()
    for metric_id in (
        "OPERATING_REVENUE_YTD",
        "PARENT_NET_PROFIT_YTD",
        "CORE_PARENT_NET_PROFIT_YTD",
        "OPERATING_CASH_FLOW_YTD",
        "TOTAL_ASSETS_END",
    ):
        if metric_id not in result.get("missing_metrics", []):
            continue
        pair = _summary_metric_pair(
            page_texts,
            metric_id,
            result=result,
            period_type=period_type,
        )
        if pair is None:
            continue
        parser._v1_4._put_metric(
            result,
            _summary_evidence(
                metric_id,
                pair[0],
                source_context=source_context,
            ),
        )
        admitted.append(metric_id)
        source_pages.add(pair[0].page_number)
    return {
        "status": (
            "PASS_DIRECT_EXPLICIT_UNIT_SUMMARY_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_DIRECT_EXPLICIT_UNIT_SUMMARY_METRICS"
        ),
        "source_pages": sorted(source_pages),
        "admitted_metric_ids": admitted,
    }


def _balance_candidates(
    result: dict[str, Any],
    page_texts: Sequence[str],
) -> list[dict[str, Any]]:
    expected_assets = _metric_cny(result, "TOTAL_ASSETS_END")
    expected_assets_unit = _metric_source_unit(result, "TOTAL_ASSETS_END")
    summary_assets_pair = _summary_metric_pair(page_texts, "TOTAL_ASSETS_END")
    summary_assets = summary_assets_pair[0].value_cny if summary_assets_pair else None
    summary_assets_unit = summary_assets_pair[0].unit if summary_assets_pair else None
    anchor_assets = expected_assets if expected_assets is not None else summary_assets
    anchor_unit = expected_assets_unit or summary_assets_unit
    compact_pages = [parser._v1_4.compact_financial_text(text) for text in page_texts]
    asset_pages = [
        index
        for index, compact in enumerate(compact_pages)
        if re.search(ASSET_PATTERNS[0], compact) is not None
    ]
    candidates: list[dict[str, Any]] = []
    for asset_page in asset_pages:
        identity_pages = _pages(asset_page, asset_page + 6, len(page_texts))
        detail_pages = _pages(asset_page - 4, asset_page + 1, len(page_texts))
        explicit_unit = _unique_explicit_unit(page_texts, identity_pages)
        explicit_scope = _nearest_balance_scope(compact_pages, asset_page)
        units = (
            tuple(parser._base.UNIT_MULTIPLIERS)
            if anchor_assets is not None
            else ((explicit_unit,) if explicit_unit is not None else ())
        )
        for unit in units:
            if explicit_unit is not None and unit != explicit_unit:
                continue
            assets = _pair(page_texts, (asset_page,), ASSET_PATTERNS, unit=unit)
            liabilities = _pair(
                page_texts,
                identity_pages,
                LIABILITY_PATTERNS,
                unit=unit,
            )
            equity = _pair(
                page_texts,
                identity_pages,
                EQUITY_PATTERNS,
                unit=unit,
            )
            total = _pair(
                page_texts,
                identity_pages,
                LIABILITY_EQUITY_TOTAL_PATTERNS,
                unit=unit,
            )
            if any(pair is None for pair in (assets, liabilities, equity, total)):
                continue
            assert assets is not None
            assert liabilities is not None
            assert equity is not None
            assert total is not None
            current_identity = _same_display(
                assets[0].value_cny,
                liabilities[0].value_cny + equity[0].value_cny,
                unit,
            ) and _same_display(
                assets[0].value_cny,
                total[0].value_cny,
                unit,
            )
            prior_identity = _same_display(
                assets[1].value_cny,
                liabilities[1].value_cny + equity[1].value_cny,
                unit,
            ) and _same_display(
                assets[1].value_cny,
                total[1].value_cny,
                unit,
            )
            summary_match = anchor_assets is not None and _same_display(
                assets[0].value_cny,
                anchor_assets,
                anchor_unit or unit,
            )
            if not current_identity or not prior_identity:
                continue
            if anchor_assets is not None and not summary_match:
                continue
            if anchor_assets is None and explicit_scope is None:
                continue
            candidates.append(
                {
                    "unit": unit,
                    "scope_anchor_kind": (
                        "SUMMARY_TOTAL_ASSETS_EXACT_DISPLAY_MATCH"
                        if expected_assets is not None and summary_match
                        else "EXPLICIT_UNIT_SUMMARY_TOTAL_ASSETS_ROUNDED_DISPLAY_MATCH"
                        if summary_match
                        else explicit_scope
                    ),
                    "asset_page": asset_page,
                    "identity_pages": identity_pages,
                    "assets": assets,
                    "liabilities": liabilities,
                    "equity": equity,
                    "total": total,
                    "receivable": _pair(
                        page_texts,
                        detail_pages,
                        RECEIVABLE_PATTERNS,
                        unit=unit,
                    ),
                    "inventory": _pair(
                        page_texts,
                        detail_pages,
                        INVENTORY_PATTERNS,
                        unit=unit,
                    ),
                }
            )
    return candidates


def _balance_identity_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["unit"],
        candidate["assets"][0].value_cny,
        candidate["liabilities"][0].value_cny,
        candidate["equity"][0].value_cny,
        candidate["total"][0].value_cny,
    )


def _add_direct_balance_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    needed = BALANCE_TARGET_METRICS.intersection(result.get("missing_metrics") or [])
    if not needed:
        return {
            "status": "NOT_NEEDED_NO_DIRECT_BALANCE_METRIC_MISSING",
            "admitted_metric_ids": [],
        }
    candidates = _balance_candidates(result, page_texts)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[_balance_identity_key(candidate)].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_DIRECT_BALANCE_IDENTITY_TUPLE_NOT_UNIQUE",
            "raw_candidate_count": len(candidates),
            "unique_identity_tuple_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = min(occurrences, key=lambda item: int(item["asset_page"]))
    unit = str(selected["unit"])
    assets = selected["assets"]
    liabilities = selected["liabilities"]
    equity = selected["equity"]
    total = selected["total"]
    validation = {
        "rule": (
            "SUMMARY_TOTAL_ASSETS_OR_EXPLICIT_CONSOLIDATED_SCOPE_PLUS_"
            "TWO_ADJACENT_COLUMNS_ASSETS_LIABILITIES_EQUITY_IDENTITY"
        ),
        "scope_anchor_kind": selected["scope_anchor_kind"],
        "summary_total_assets_cny": str(
            _metric_cny(result, "TOTAL_ASSETS_END")
        ),
        "unit_uniquely_selected": True,
        "unit": unit,
        "raw_candidate_count": len(candidates),
        "identical_statement_occurrence_count": len(occurrences),
        "unique_identity_tuple_count": 1,
        "first_column_identity_passed": True,
        "second_column_identity_passed": True,
        "total_assets_current_cny": str(assets[0].value_cny),
        "total_liabilities_current_cny": str(liabilities[0].value_cny),
        "total_equity_current_cny": str(equity[0].value_cny),
        "liabilities_equity_total_current_cny": str(total[0].value_cny),
        "total_assets_prior_cny": str(assets[1].value_cny),
        "total_liabilities_prior_cny": str(liabilities[1].value_cny),
        "total_equity_prior_cny": str(equity[1].value_cny),
        "liabilities_equity_total_prior_cny": str(total[1].value_cny),
    }
    metric_pairs: dict[str, Any] = {
        "TOTAL_ASSETS_END": assets,
        "TOTAL_LIABILITIES_END": liabilities,
    }
    for metric_id, candidate_key in (
        ("ACCOUNTS_RECEIVABLE_END", "receivable"),
        ("INVENTORY_END", "inventory"),
    ):
        pairs = [
            occurrence[candidate_key]
            for occurrence in occurrences
            if occurrence[candidate_key] is not None
        ]
        unique_pairs = {pair[0].value_cny: pair for pair in pairs}
        if len(unique_pairs) == 1:
            metric_pairs[metric_id] = next(iter(unique_pairs.values()))
    admitted: list[str] = []
    for metric_id in (
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
    ):
        pair = metric_pairs.get(metric_id)
        if metric_id not in result.get("missing_metrics", []) or pair is None:
            continue
        parser._v1_4._put_metric(
            result,
            _direct_evidence(
                metric_id,
                pair[0],
                value_period_scope="PERIOD_END",
                section_name="DIRECT_RECONCILED_CONSOLIDATED_BALANCE_SHEET",
                validation=validation,
                source_context=source_context,
            ),
        )
        admitted.append(metric_id)
    return {
        "status": (
            "PASS_DIRECT_BALANCE_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_DIRECT_BALANCE_DETAILS_NOT_UNIQUE"
        ),
        "source_pages": sorted(
            {
                candidate[0].page_number
                for candidate in metric_pairs.values()
                if candidate is not None
            }
        ),
        "source_unit": unit,
        "raw_candidate_count": len(candidates),
        "identical_statement_occurrence_count": len(occurrences),
        "admitted_metric_ids": admitted,
    }


def _add_direct_receivable_note_reconciliation(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    if "ACCOUNTS_RECEIVABLE_END" not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_ACCOUNTS_RECEIVABLE_ALREADY_PRESENT",
            "admitted_metric_ids": [],
        }
    balance_candidates = _balance_candidates(result, page_texts)
    grouped_balance: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in balance_candidates:
        grouped_balance[_balance_identity_key(candidate)].append(candidate)
    if len(grouped_balance) != 1:
        return {
            "status": "NO_MATCH_DIRECT_NOTE_BALANCE_SCOPE_NOT_UNIQUE",
            "balance_candidate_count": len(balance_candidates),
            "unique_balance_identity_count": len(grouped_balance),
            "admitted_metric_ids": [],
        }
    balance_occurrences = next(iter(grouped_balance.values()))
    combined_pairs: dict[tuple[Decimal, Decimal], Any] = {}
    for balance in balance_occurrences:
        combined = _pair(
            page_texts,
            _pages(
                int(balance["asset_page"]) - 4,
                int(balance["asset_page"]) + 1,
                len(page_texts),
            ),
            (r"应收票据(?:及|和)应收账款",),
            unit=str(balance["unit"]),
        )
        if combined is not None:
            combined_pairs[_pair_values(combined)] = combined
    if len(combined_pairs) != 1:
        return {
            "status": "NO_MATCH_DIRECT_NOTE_COMBINED_RECEIVABLE_NOT_UNIQUE",
            "combined_pair_count": len(combined_pairs),
            "admitted_metric_ids": [],
        }
    combined = next(iter(combined_pairs.values()))
    selected_balance = min(
        balance_occurrences,
        key=lambda candidate: int(candidate["asset_page"]),
    )
    unit = str(selected_balance["unit"])
    first_note_page = int(selected_balance["asset_page"]) + 1
    consolidated_note_end = _consolidated_note_end_page(
        page_texts,
        first_note_page=first_note_page,
    )
    bills_by_page: dict[int, Any] = {}
    receivable_by_page: dict[int, Any] = {}
    for page_index in range(first_note_page, consolidated_note_end):
        bills = _pair(
            page_texts,
            (page_index,),
            (r"(?<!及)(?<!和)应收票据(?!及应收账款)(?!和应收账款)",),
            unit=unit,
        ) or _cross_page_bills_classification_total_pair(
            page_texts[page_index - 1] if page_index > 0 else "",
            page_texts[page_index],
            page_number=page_index + 1,
            unit=unit,
        )
        receivable = _receivable_classification_total_pair(
            page_texts[page_index],
            page_number=page_index + 1,
            unit=unit,
        ) or _pair(
            page_texts,
            (page_index,),
            (
                r"(?<!票据及)(?<!票据和)(?<!其他)应收账款"
                r"(?!及应收票据)(?!和应收票据)"
                r"(?!及合同资产)(?!和合同资产)(?!总体)(?!按)(?!主要)(?!的)",
            ),
            unit=unit,
        )
        if bills is not None:
            bills_by_page[page_index] = bills
        if receivable is not None:
            receivable_by_page[page_index] = receivable

    reconciled: list[dict[str, Any]] = []
    for receivable_page, receivable in receivable_by_page.items():
        compact_window = parser._v1_4.compact_financial_text(
            receivable[0].line_window
        )
        direct_current = _same_display(
            combined[0].value_cny,
            receivable[0].value_cny,
            unit,
        )
        direct_second = _same_display(
            combined[1].value_cny,
            receivable[1].value_cny,
            unit,
        )
        explicit_zero_bills = any(
            marker in parser._v1_4.compact_financial_text(page_texts[receivable_page])
            for marker in (
                "应收票据为0",
                "应收票据余额为0",
                "应收票据期末余额为0",
            )
        )
        if direct_current and (
            "合并列示为" not in compact_window
            and "合并计入" not in compact_window
            and "并入" not in compact_window
            or explicit_zero_bills
        ):
            reconciled.append(
                {
                    "receivable": receivable,
                    "bills": None,
                    "receivable_page": receivable_page,
                    "bills_page": None,
                    "route": "NET_RECEIVABLE_EQUALS_COMBINED_TOTAL",
                    "current_identity_passed": True,
                    "second_column_identity_passed": direct_second,
                    "explicit_zero_bills": explicit_zero_bills,
                }
            )
        for bills_page, bills in bills_by_page.items():
            if abs(receivable_page - bills_page) > 3:
                continue
            current_pass = _same_display(
                combined[0].value_cny,
                bills[0].value_cny + receivable[0].value_cny,
                unit,
            )
            second_pass = _same_display(
                combined[1].value_cny,
                bills[1].value_cny + receivable[1].value_cny,
                unit,
            )
            if not current_pass:
                continue
            reconciled.append(
                {
                    "receivable": receivable,
                    "bills": bills,
                    "receivable_page": receivable_page,
                    "bills_page": bills_page,
                    "route": "BILLS_PLUS_NET_RECEIVABLE_EQUALS_COMBINED_TOTAL",
                    "current_identity_passed": True,
                    "second_column_identity_passed": second_pass,
                    "explicit_zero_bills": False,
                }
            )
    unique_values: dict[Decimal, list[dict[str, Any]]] = defaultdict(list)
    for candidate in reconciled:
        unique_values[candidate["receivable"][0].value_cny].append(candidate)
    if len(unique_values) != 1:
        return {
            "status": "NO_MATCH_DIRECT_NOTE_RECEIVABLE_VALUE_NOT_UNIQUE",
            "reconciled_candidate_count": len(reconciled),
            "unique_current_value_count": len(unique_values),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(unique_values.values()))
    selected = min(
        occurrences,
        key=lambda candidate: (
            int(candidate["receivable_page"]),
            int(candidate["bills_page"] or -1),
        ),
    )
    receivable = selected["receivable"]
    bills = selected["bills"]
    validation = {
        "rule": selected["route"],
        "statement_combined_current_cny": str(combined[0].value_cny),
        "statement_combined_second_column_cny": str(combined[1].value_cny),
        "net_receivable_current_cny": str(receivable[0].value_cny),
        "net_receivable_second_column_cny": str(receivable[1].value_cny),
        "bills_current_cny": str(bills[0].value_cny) if bills else "0",
        "bills_second_column_cny": (
            str(bills[1].value_cny) if bills else "0"
        ),
        "current_column_identity_passed": True,
        "second_column_identity_passed": selected[
            "second_column_identity_passed"
        ],
        "explicit_zero_bills": selected["explicit_zero_bills"],
        "identical_current_value_occurrence_count": len(occurrences),
        "unique_current_value_count": 1,
    }
    parser._v1_4._put_metric(
        result,
        _direct_evidence(
            "ACCOUNTS_RECEIVABLE_END",
            receivable[0],
            value_period_scope="PERIOD_END",
            section_name="CONSOLIDATED_NOTES_RECEIVABLE_BREAKDOWN",
            validation=validation,
            source_context=source_context,
        ),
    )
    return {
        "status": "PASS_DIRECT_NOTE_RECEIVABLE_RECONCILED_AND_ADMITTED",
        "route": selected["route"],
        "statement_combined_source_page": combined[0].page_number,
        "note_receivable_source_page": receivable[0].page_number,
        "note_bills_source_page": bills[0].page_number if bills else None,
        "current_column_identity_passed": True,
        "second_column_identity_passed": selected["second_column_identity_passed"],
        "admitted_metric_ids": ["ACCOUNTS_RECEIVABLE_END"],
    }


def _receivable_classification_total_pair(
    page_text: str,
    *,
    page_number: int,
    unit: str,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    """读取应收账款分类披露表“合计”行的当期、期初账面价值。"""

    compact = parser._v1_4.compact_financial_text(page_text)
    if "应收账款分类披露" not in compact or "账面价值" not in compact:
        return None
    amount = r"\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?"
    pattern = re.compile(
        rf"合计\s*(?P<gross_current>{amount})\s*/\s*"
        rf"(?P<allowance_current>{amount})\s*/\s*"
        rf"(?P<net_current>{amount})\s+"
        rf"(?P<gross_prior>{amount})\s*/\s*"
        rf"(?P<allowance_prior>{amount})\s*/\s*"
        rf"(?P<net_prior>{amount})"
    )
    matches = list(pattern.finditer(page_text))
    if len(matches) != 1:
        return None
    match = matches[0]
    selected: list[parser._base.ParsedCandidate] = []
    for amount_index, group_name in ((2, "net_current"), (5, "net_prior")):
        raw_value = str(match.group(group_name))
        start, end = match.span(group_name)
        selected.append(
            parser._base.ParsedCandidate(
                value_cny=(
                    parser._base.parse_decimal(raw_value)
                    * parser._base.UNIT_MULTIPLIERS[unit]
                ),
                page_number=page_number,
                label="应收账款分类披露—合计账面价值",
                raw_value=raw_value,
                unit=unit,
                line_window=page_text[
                    max(0, match.start() - 500) : min(len(page_text), match.end() + 180)
                ],
                match_span=match.span(),
                value_span=(start, end),
                selected_amount_index=amount_index,
            )
        )
    return selected[0], selected[1]


def _cross_page_bills_classification_total_pair(
    previous_page_text: str,
    page_text: str,
    *,
    page_number: int,
    unit: str,
) -> tuple[parser._base.ParsedCandidate, parser._base.ParsedCandidate] | None:
    previous_compact = parser._v1_4.compact_financial_text(previous_page_text)
    current_compact = parser._v1_4.compact_financial_text(page_text[:1_500])
    if "应收票据分类列示" not in previous_compact:
        return None
    if "项目期末余额期初余额" not in current_compact:
        return None
    section_end = len(page_text)
    for marker in ("(2).", "（2）", "2)."):
        position = page_text.find(marker)
        if position >= 0:
            section_end = min(section_end, position)
    section = page_text[:section_end]
    amount = r"\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?"
    matches = list(
        re.finditer(
            rf"合计\s*(?P<current>{amount})\s+(?P<prior>{amount})",
            section,
        )
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    selected: list[parser._base.ParsedCandidate] = []
    for amount_index, group_name in ((0, "current"), (1, "prior")):
        raw_value = str(match.group(group_name))
        start, end = match.span(group_name)
        selected.append(
            parser._base.ParsedCandidate(
                value_cny=(
                    parser._base.parse_decimal(raw_value)
                    * parser._base.UNIT_MULTIPLIERS[unit]
                ),
                page_number=page_number,
                label="应收票据分类列示—合计",
                raw_value=raw_value,
                unit=unit,
                line_window=(
                    previous_page_text[-400:]
                    + f"\n[相邻页表格续接：PDF第{page_number}页]"
                    + section[: min(len(section), match.end() + 180)]
                ),
                match_span=match.span(),
                value_span=(start, end),
                selected_amount_index=amount_index,
            )
        )
    return selected[0], selected[1]


def _consolidated_note_end_page(
    page_texts: Sequence[str],
    *,
    first_note_page: int,
) -> int:
    for page_index in range(first_note_page, len(page_texts)):
        compact = parser._v1_4.compact_financial_text(page_texts[page_index])
        if any(
            marker in compact
            for marker in (
                "母公司财务报表主要项目注释",
                "母公司财务报表项目注释",
                "公司财务报表主要项目注释",
            )
        ):
            return page_index
    return len(page_texts)


def _q3_amount_index(
    page_texts: Sequence[str],
    page_index: int,
    period_type: str | None,
) -> tuple[int | None, dict[str, bool]]:
    if str(period_type or "").upper() != "Q3":
        return 0, {
            "q3_ytd_context_present": False,
            "q3_current_quarter_context_present": False,
        }
    context = "".join(
        parser._v1_4.compact_financial_text(page_texts[index])
        for index in _pages(page_index - 4, page_index + 2, len(page_texts))
    )
    has_ytd = any(marker in context for marker in parser._v1_6._v1_3.YTD_MARKERS)
    has_quarter = any(
        marker in context for marker in parser._v1_6._v1_3.CURRENT_QUARTER_MARKERS
    )
    if not has_ytd:
        return None, {
            "q3_ytd_context_present": False,
            "q3_current_quarter_context_present": has_quarter,
        }
    return (2 if has_quarter else 0), {
        "q3_ytd_context_present": True,
        "q3_current_quarter_context_present": has_quarter,
    }


def _income_candidates(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> list[dict[str, Any]]:
    expected_parent = _metric_cny(result, "PARENT_NET_PROFIT_YTD")
    expected_revenue = _metric_cny(result, "OPERATING_REVENUE_YTD")
    expected_parent_unit = _metric_source_unit(result, "PARENT_NET_PROFIT_YTD")
    expected_revenue_unit = _metric_source_unit(result, "OPERATING_REVENUE_YTD")
    if expected_parent is None and expected_revenue is None:
        return []
    compact_pages = [parser._v1_4.compact_financial_text(text) for text in page_texts]
    operating_pages = [
        index
        for index, compact in enumerate(compact_pages)
        if "营业利润" in compact
    ]
    candidates: list[dict[str, Any]] = []
    for operating_page in operating_pages:
        amount_index, q3_context = _q3_amount_index(
            page_texts,
            operating_page,
            period_type,
        )
        if amount_index is None:
            continue
        section_pages = _pages(operating_page - 2, operating_page + 4, len(page_texts))
        explicit_unit = _unique_explicit_unit(page_texts, section_pages)
        for unit in parser._base.UNIT_MULTIPLIERS:
            if explicit_unit is not None and unit != explicit_unit:
                continue
            operating_profit = _whole_page_pair(
                page_texts,
                (operating_page,),
                OPERATING_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            if operating_profit is None:
                continue
            revenue = _whole_page_pair(
                page_texts,
                section_pages,
                REVENUE_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            net_profit = _whole_page_pair(
                page_texts,
                section_pages,
                NET_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            parent_profit = _whole_page_pair(
                page_texts,
                section_pages,
                PARENT_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            minority_profit = _whole_page_pair(
                page_texts,
                section_pages,
                MINORITY_PROFIT_PATTERNS,
                unit=unit,
                amount_index=amount_index,
            )
            parent_match = (
                expected_parent is not None
                and parent_profit is not None
                and _same_display(
                    parent_profit[0].value_cny,
                    expected_parent,
                    expected_parent_unit or unit,
                )
            )
            revenue_match = (
                expected_revenue is not None
                and revenue is not None
                and _same_display(
                    revenue[0].value_cny,
                    expected_revenue,
                    expected_revenue_unit or unit,
                )
            )
            if expected_parent is not None and parent_profit is not None and not parent_match:
                continue
            if expected_revenue is not None and revenue is not None and not revenue_match:
                continue
            if not parent_match and not revenue_match:
                continue
            parent_identity_current: bool | None = None
            parent_identity_prior: bool | None = None
            if all(pair is not None for pair in (net_profit, parent_profit, minority_profit)):
                assert net_profit is not None
                assert parent_profit is not None
                assert minority_profit is not None
                parent_identity_current = _same_display(
                    net_profit[0].value_cny,
                    parent_profit[0].value_cny + minority_profit[0].value_cny,
                    unit,
                )
                parent_identity_prior = _same_display(
                    net_profit[1].value_cny,
                    parent_profit[1].value_cny + minority_profit[1].value_cny,
                    unit,
                )
                if not parent_identity_current or not parent_identity_prior:
                    continue
            candidates.append(
                {
                    "unit": unit,
                    "amount_index": amount_index,
                    "q3_context": q3_context,
                    "operating_profit": operating_profit,
                    "revenue": revenue,
                    "net_profit": net_profit,
                    "parent_profit": parent_profit,
                    "minority_profit": minority_profit,
                    "parent_anchor_match": parent_match,
                    "revenue_anchor_match": revenue_match,
                    "parent_identity_current": parent_identity_current,
                    "parent_identity_prior": parent_identity_prior,
                }
            )
    return candidates


def _income_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["unit"],
        candidate["amount_index"],
        candidate["operating_profit"][0].value_cny,
    )


def _add_direct_income_metrics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    needed = INCOME_TARGET_METRICS.intersection(result.get("missing_metrics") or [])
    if not needed:
        return {
            "status": "NOT_NEEDED_NO_DIRECT_INCOME_METRIC_MISSING",
            "admitted_metric_ids": [],
        }
    candidates = _income_candidates(result, page_texts, period_type=period_type)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[_income_key(candidate)].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_DIRECT_INCOME_FACT_TUPLE_NOT_UNIQUE",
            "raw_candidate_count": len(candidates),
            "unique_fact_tuple_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = min(
        occurrences,
        key=lambda item: int(item["operating_profit"][0].page_number),
    )
    unit = str(selected["unit"])
    validation = {
        "rule": (
            "SUMMARY_PARENT_PROFIT_OR_REVENUE_ANCHOR_PLUS_UNIQUE_UNIT_"
            "AND_DUAL_PERIOD_INCOME_COLUMNS"
        ),
        "summary_parent_profit_cny": str(
            _metric_cny(result, "PARENT_NET_PROFIT_YTD")
        ),
        "summary_revenue_cny": str(_metric_cny(result, "OPERATING_REVENUE_YTD")),
        "parent_profit_anchor_match_passed": selected["parent_anchor_match"],
        "revenue_anchor_match_passed": selected["revenue_anchor_match"],
        "parent_plus_minority_identity_current": selected[
            "parent_identity_current"
        ],
        "parent_plus_minority_identity_prior": selected["parent_identity_prior"],
        "unit_uniquely_selected": True,
        "unit": unit,
        "selected_amount_index": selected["amount_index"],
        **selected["q3_context"],
        "raw_candidate_count": len(candidates),
        "identical_statement_occurrence_count": len(occurrences),
        "unique_fact_tuple_count": 1,
    }
    metric_pairs: dict[str, Any] = {
        "OPERATING_PROFIT_YTD": selected["operating_profit"],
    }
    for metric_id, candidate_key in (
        ("OPERATING_REVENUE_YTD", "revenue"),
        ("PARENT_NET_PROFIT_YTD", "parent_profit"),
    ):
        pairs = [
            occurrence[candidate_key]
            for occurrence in occurrences
            if occurrence[candidate_key] is not None
        ]
        unique_pairs = {pair[0].value_cny: pair for pair in pairs}
        if len(unique_pairs) == 1:
            metric_pairs[metric_id] = next(iter(unique_pairs.values()))
    admitted: list[str] = []
    for metric_id in (
        "OPERATING_REVENUE_YTD",
        "OPERATING_PROFIT_YTD",
        "PARENT_NET_PROFIT_YTD",
    ):
        pair = metric_pairs.get(metric_id)
        if metric_id not in result.get("missing_metrics", []) or pair is None:
            continue
        parser._v1_4._put_metric(
            result,
            _direct_evidence(
                metric_id,
                pair[0],
                value_period_scope="YEAR_TO_DATE",
                section_name="DIRECT_RECONCILED_CONSOLIDATED_INCOME_STATEMENT",
                validation=validation,
                source_context=source_context,
            ),
        )
        admitted.append(metric_id)
    return {
        "status": (
            "PASS_DIRECT_INCOME_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_DIRECT_INCOME_DETAILS_NOT_UNIQUE"
        ),
        "source_pages": sorted(
            {
                pair[0].page_number
                for pair in metric_pairs.values()
                if pair is not None
            }
        ),
        "source_unit": unit,
        "selected_amount_index": selected["amount_index"],
        "raw_candidate_count": len(candidates),
        "identical_statement_occurrence_count": len(occurrences),
        "admitted_metric_ids": admitted,
    }


def _add_manual_direct_text_receivable_reconciliation_fact(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    announcement_id = str(source_context.get("announcement_id") or "")
    fact = MANUAL_DIRECT_TEXT_RECEIVABLE_RECONCILIATION_FACTS.get(announcement_id)
    if fact is None:
        return {
            "status": "NOT_APPLICABLE_NO_MANUAL_DIRECT_TEXT_RECEIVABLE_FACT",
            "admitted_metric_ids": [],
        }
    metric_id = "ACCOUNTS_RECEIVABLE_END"
    if metric_id not in (result.get("missing_metrics") or []):
        return {
            "status": "NOT_NEEDED_MANUAL_DIRECT_TEXT_RECEIVABLE_PRESENT",
            "admitted_metric_ids": [],
        }

    page_number = int(fact["page_number"])
    if page_number < 1 or page_number > len(page_texts):
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_RECEIVABLE_PAGE_OUT_OF_RANGE",
            "admitted_metric_ids": [],
        }
    page_text = str(page_texts[page_number - 1])
    compact = parser._v1_4.compact_financial_text(page_text)
    if (
        "应收账款" not in compact
        or "坏账准备" not in compact
        or not any(
            label in compact
            for label in ("应收票据及应收账款", "应收票据和应收账款")
        )
    ):
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_RECEIVABLE_LABELS_NOT_FOUND",
            "admitted_metric_ids": [],
        }

    unit = str(fact["unit"])
    unit_patterns = (
        rf"(?:金额单位|单位)(?:均?为)?[：:]?(?:人民币)?{re.escape(unit)}",
        rf"人民币{re.escape(unit)}",
    )
    if not any(re.search(pattern, compact) for pattern in unit_patterns):
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_RECEIVABLE_UNIT_NOT_FOUND",
            "admitted_metric_ids": [],
        }

    component_names = (
        "current_gross",
        "current_allowance",
        "current_bills",
        "current_combined",
        "prior_gross",
        "prior_allowance",
        "prior_bills",
        "prior_combined",
    )
    component_raw = {name: str(fact[name]) for name in component_names}
    missing_components: list[str] = []
    for name, raw_value in component_raw.items():
        digits = "".join(re.findall(r"\d", raw_value))
        digit_pattern = r"[\s,，.]*".join(re.escape(char) for char in digits)
        if re.search(rf"(?<!\d){digit_pattern}(?!\d)", page_text) is None:
            missing_components.append(name)
    if missing_components:
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_RECEIVABLE_COMPONENTS_NOT_FOUND",
            "missing_components": missing_components,
            "admitted_metric_ids": [],
        }

    components = {name: Decimal(value) for name, value in component_raw.items()}
    current_net = components["current_gross"] - components["current_allowance"]
    prior_net = components["prior_gross"] - components["prior_allowance"]
    current_identity_passed = (
        current_net + components["current_bills"] == components["current_combined"]
    )
    prior_identity_passed = (
        prior_net + components["prior_bills"] == components["prior_combined"]
    )
    if (
        current_net < 0
        or prior_net < 0
        or not current_identity_passed
        or not prior_identity_passed
    ):
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_RECEIVABLE_IDENTITIES_FAILED",
            "current_identity_passed": current_identity_passed,
            "prior_identity_passed": prior_identity_passed,
            "admitted_metric_ids": [],
        }

    multiplier = parser._base.UNIT_MULTIPLIERS[unit]
    locator = {
        "page": page_number,
        "section": (
            "MANUAL_EXACT_OFFICIAL_PDF_DIRECT_TEXT_RECEIVABLE_NOTE_"
            "RECONCILIATION"
        ),
        "parser_version": DIRECT_PARSER_VERSION,
        "components": component_raw,
        "derived_current_net_receivable": str(current_net),
        "derived_prior_net_receivable": str(prior_net),
        "validation": {
            "official_pdf_direct_text_page_exact": True,
            "receivable_note_labels_present": True,
            "all_component_values_present": True,
            "current_gross_less_allowance_equals_net_receivable": True,
            "prior_gross_less_allowance_equals_net_receivable": True,
            "current_bills_plus_net_receivable_equals_combined_total": True,
            "prior_bills_plus_net_receivable_equals_combined_total": True,
            "unit_contract_exact": True,
        },
        "source_context": source_context,
    }
    parser._v1_4._put_metric(
        result,
        parser._base.MetricEvidence(
            metric_id=metric_id,
            metric_value_cny=float(current_net * Decimal(str(multiplier))),
            statement_scope="CONSOLIDATED_ONLY",
            value_period_scope="PERIOD_END",
            source_page=page_number,
            source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
            source_label="应收账款（应收票据及应收账款附注，扣除坏账准备）",
            source_raw_value=(
                f"{components['current_gross']:,} - "
                f"{components['current_allowance']:,} = {current_net:,}"
            ),
            source_unit=unit,
            source_unit_multiplier=float(multiplier),
            source_method=(
                "OFFICIAL_PDF_DIRECT_TEXT_RECEIVABLE_NOTE_DUAL_ACCOUNTING_"
                "IDENTITY_RECONCILED"
            ),
            verification_status=DIRECT_VERIFICATION_STATUS,
        ),
    )
    return {
        "status": "PASS_MANUAL_DIRECT_TEXT_RECEIVABLE_RECONCILED_AND_ADMITTED",
        "source_page": page_number,
        "source_unit": unit,
        "current_net_receivable": str(current_net),
        "prior_net_receivable": str(prior_net),
        "current_identity_passed": True,
        "prior_identity_passed": True,
        "admitted_metric_ids": [metric_id],
    }


def _add_single_manual_direct_text_row_fact(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    fact: dict[str, Any],
    source_context: dict[str, Any],
) -> dict[str, Any]:
    announcement_id = str(source_context.get("announcement_id") or "")
    metric_id = str(fact["metric_id"])
    if metric_id not in (result.get("missing_metrics") or []):
        return {
            "status": "NOT_NEEDED_MANUAL_DIRECT_TEXT_METRIC_PRESENT",
            "admitted_metric_ids": [],
        }

    page_number = int(fact["page_number"])
    if page_number < 1 or page_number > len(page_texts):
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_PAGE_OUT_OF_RANGE",
            "admitted_metric_ids": [],
        }
    compact = parser._v1_4.compact_financial_text(page_texts[page_number - 1])
    source_section_label = str(fact.get("source_section_label") or "")
    if source_section_label:
        compact_section_label = parser._v1_4.compact_financial_text(source_section_label)
        section_start = compact.find(compact_section_label)
        if section_start < 0:
            return {
                "status": "REJECT_MANUAL_DIRECT_TEXT_SECTION_LABEL_NOT_FOUND",
                "admitted_metric_ids": [],
            }
        compact = compact[section_start:]

    label = str(fact["label"])
    label_pattern = str(
        fact.get("label_pattern")
        or re.escape(parser._v1_4.compact_financial_text(label))
    )
    label_matches = list(re.finditer(label_pattern, compact))
    if len(label_matches) != 1:
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_LABEL_NOT_UNIQUE",
            "label_match_count": len(label_matches),
            "admitted_metric_ids": [],
        }
    label_match = label_matches[0]
    row_window = compact[label_match.start() : label_match.end() + 600]
    value_search_scope = str(fact.get("value_search_scope") or "ROW_AFTER_LABEL")
    if value_search_scope not in {"ROW_AFTER_LABEL", "PAGE"}:
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_VALUE_SEARCH_SCOPE_INVALID",
            "admitted_metric_ids": [],
        }
    value_window = compact if value_search_scope == "PAGE" else row_window

    unit = str(fact["unit"])
    unit_page_number = int(fact.get("unit_page_number") or page_number)
    if unit_page_number < 1 or unit_page_number > len(page_texts):
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_UNIT_PAGE_OUT_OF_RANGE",
            "admitted_metric_ids": [],
        }
    unit_compact = parser._v1_4.compact_financial_text(
        page_texts[unit_page_number - 1]
    )
    unit_patterns = (
        rf"(?:金额单位|单位)[：:]?(?:人民币)?{re.escape(unit)}",
        rf"人民币{re.escape(unit)}",
    )
    if not any(re.search(pattern, unit_compact) for pattern in unit_patterns):
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_UNIT_NOT_FOUND",
            "admitted_metric_ids": [],
        }

    current_raw = str(fact["current"])
    prior_raw = str(fact["prior"])
    current_digits = "".join(re.findall(r"\d", current_raw))
    prior_digits = "".join(re.findall(r"\d", prior_raw))
    digit_stream = "".join(re.findall(r"\d", value_window))
    if current_digits not in digit_stream or prior_digits not in digit_stream:
        return {
            "status": "REJECT_MANUAL_DIRECT_TEXT_DUAL_PERIOD_VALUES_NOT_FOUND",
            "admitted_metric_ids": [],
        }

    for raw_value in (current_raw, prior_raw):
        if not raw_value.startswith("-"):
            continue
        digits = "".join(re.findall(r"\d", raw_value))
        digit_pattern = r"[\s,，.]*".join(re.escape(char) for char in digits)
        negative_pattern = rf"(?:[-−]\s*{digit_pattern}|[（(]\s*{digit_pattern}\s*[）)])"
        if re.search(negative_pattern, value_window) is None:
            return {
                "status": "REJECT_MANUAL_DIRECT_TEXT_NEGATIVE_SIGN_NOT_CONFIRMED",
                "admitted_metric_ids": [],
            }

    multiplier = parser._base.UNIT_MULTIPLIERS[unit]
    current = Decimal(current_raw)
    prior = Decimal(prior_raw)
    locator = {
        "page": page_number,
        "section": "MANUAL_EXACT_OFFICIAL_PDF_DIRECT_TEXT_ROW",
        "parser_version": DIRECT_PARSER_VERSION,
        "source_section_label": source_section_label or None,
        "unit_page": unit_page_number,
        "current_raw_value": current_raw,
        "prior_raw_value": prior_raw,
        "value_search_scope": value_search_scope,
        "validation": {
            "official_pdf_direct_text_page_exact": True,
            "label_unique_in_selected_scope": True,
            "dual_period_values_present_in_frozen_search_scope": True,
            "negative_signs_or_parentheses_confirmed": True,
            "unit_contract_exact": True,
        },
        "source_context": source_context,
    }
    parser._v1_4._put_metric(
        result,
        parser._base.MetricEvidence(
            metric_id=metric_id,
            metric_value_cny=float(current * Decimal(str(multiplier))),
            statement_scope="CONSOLIDATED_ONLY",
            value_period_scope=str(fact["value_period_scope"]),
            source_page=page_number,
            source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
            source_label=label,
            source_raw_value=f"{current:,}",
            source_unit=unit,
            source_unit_multiplier=float(multiplier),
            source_method="OFFICIAL_PDF_DIRECT_TEXT_MANUAL_EXACT_DUAL_PERIOD_ROW",
            verification_status=DIRECT_VERIFICATION_STATUS,
        ),
    )
    return {
        "status": "PASS_MANUAL_DIRECT_TEXT_ROW_FACT_ADMITTED",
        "source_page": page_number,
        "source_unit": unit,
        "current_value": str(current),
        "prior_value": str(prior),
        "admitted_metric_ids": [metric_id],
    }


def _add_manual_direct_text_row_fact(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    announcement_id = str(source_context.get("announcement_id") or "")
    entry = MANUAL_DIRECT_TEXT_ROW_FACTS.get(announcement_id)
    if entry is None:
        return {
            "status": "NOT_APPLICABLE_NO_MANUAL_DIRECT_TEXT_ROW_FACT",
            "admitted_metric_ids": [],
        }
    facts = entry if isinstance(entry, tuple) else (entry,)
    receipts = [
        _add_single_manual_direct_text_row_fact(
            result,
            page_texts,
            fact=fact,
            source_context=source_context,
        )
        for fact in facts
    ]
    if len(receipts) == 1:
        return receipts[0]
    admitted_metric_ids = [
        metric_id
        for receipt in receipts
        for metric_id in receipt.get("admitted_metric_ids", [])
    ]
    return {
        "status": (
            "PASS_MANUAL_DIRECT_TEXT_ROW_FACTS_ADMITTED"
            if admitted_metric_ids
            else "NO_MANUAL_DIRECT_TEXT_ROW_FACTS_ADMITTED"
        ),
        "fact_receipts": receipts,
        "admitted_metric_ids": admitted_metric_ids,
    }


def add_direct_pdf_document_statistics(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    normalized = [
        parser._v1_4.normalize_financial_text(page_text) for page_text in page_texts
    ]
    before_missing = list(result.get("missing_metrics") or [])
    balance_receipt = _add_direct_balance_metrics(
        result,
        normalized,
        source_context=source_context,
    )
    receivable_note_receipt = _add_direct_receivable_note_reconciliation(
        result,
        normalized,
        source_context=source_context,
    )
    manual_direct_text_receivable_receipt = (
        _add_manual_direct_text_receivable_reconciliation_fact(
            result,
            normalized,
            source_context=source_context,
        )
    )
    summary_receipt = _add_direct_summary_metrics(
        result,
        normalized,
        period_type=period_type,
        source_context=source_context,
    )
    income_receipt = _add_direct_income_metrics(
        result,
        normalized,
        period_type=period_type,
        source_context=source_context,
    )
    manual_direct_text_row_receipt = _add_manual_direct_text_row_fact(
        result,
        normalized,
        source_context=source_context,
    )
    parser._restamp_result(result)
    after_missing = list(result.get("missing_metrics") or [])
    admitted = [
        metric_id for metric_id in before_missing if metric_id not in after_missing
    ]
    return {
        "status": (
            "PASS_DIRECT_PDF_DOCUMENT_STATISTICS_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_DIRECT_PDF_DOCUMENT_STATISTICS"
        ),
        "direct_parser_version": DIRECT_PARSER_VERSION,
        "balance": balance_receipt,
        "receivable_note_reconciliation": receivable_note_receipt,
        "manual_direct_text_receivable_reconciliation": (
            manual_direct_text_receivable_receipt
        ),
        "summary": summary_receipt,
        "income": income_receipt,
        "manual_direct_text_row": manual_direct_text_row_receipt,
        "admitted_metric_ids": admitted,
        "remaining_missing_metrics": after_missing,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
