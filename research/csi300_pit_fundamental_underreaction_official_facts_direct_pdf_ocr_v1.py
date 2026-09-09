from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
import json
import re
from typing import Any, Iterable, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_v1 as direct
from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser


DIRECT_OCR_VERSION = "CSI300_PIT_OFFICIAL_FACTS_TARGETED_DIRECT_PDF_OCR_V1_3_86"
DIRECT_OCR_VERIFICATION_STATUS = (
    "PASS_OFFICIAL_PDF_TARGETED_DUAL_RENDER_OCR_ACCOUNTING_IDENTITY_RECONCILED"
)
DIRECT_OCR_RENDER_SCALES = (2.25, 3.75, 4.5)

# 少数官方 PDF 使用无法从文本层识别的嵌入字体，自动目录/标题锚点会失效。
# 页码只负责把 OCR 限定到肉眼可确认的财务报表区；数值仍需通过多尺度一致性与
# 冻结会计恒等式后才能准入。页码为 PDF 阅读器显示的 1 起始页码。
MANUAL_FINANCIAL_STATEMENT_PAGE_NUMBERS: dict[str, tuple[int, ...]] = {
    "1202251607": (3, 11, 12, 13, 16, 19),
    "1202268049": (4,),
    "1202577310": tuple(range(50, 63)),
    "1202636649": (28,),
    "1202658580": (59,),
    "1202789596": (4, 17, 18, 19, 21, 22),
    "1202791853": (3, 11, 12, 13, 18, 22),
    "1202795561": (12, 13),
    "1203242141": (108,),
    "1203374405": (8, 10),
    "1203416615": (11, 12),
    "1203848634": (6,) + tuple(range(41, 52)),
    "1204081143": (12, 13, 15),
    "1204555388": (125,),
    "1204682861": (108, 109),
    "1204803034": (9, 10, 12),
    "1205334041": (5, 53, 54, 57, 59, 78),
    "1205361911": (112,),
    "1205363576": tuple(range(63, 79)),
    "1205968967": (112, 113, 165),
    "1205969212": (133, 134, 137),
    "1206869181": (67,),
    "1207036153": (12,),
    "1207047784": (11, 12, 13),
    "1207441457": (119, 120),
    "1207423623": (79, 80),
    "1207432718": (89, 90, 92),
    "1207586412": (83, 84, 87),
    "1207641522": (135,),
    "1207666500": (12, 14),
    "1207684809": (11, 12, 13),
    "1207688733": (9, 10, 12),
    "1207691674": (18,),
    "1208326876": (76,),
    "1208321741": tuple(range(59, 75)),
    "1208345210": tuple(range(49, 59)),
    "1208627222": (34,),
    "1208637839": (27, 28, 29),
    "1208662817": (11, 12, 14),
    "1208664350": (13, 14, 15),
    "1209464252": (98, 99, 100),
    "1209477137": (121, 122),
    "1209481072": (92,),
    "1209490194": (86,),
    "1209495189": (130, 131),
    "1209726071": (144, 145),
    "1209870320": (11, 12, 13),
    "1210899453": tuple(range(63, 79)),
    "1210899621": (51,),
    "1210900865": (7,),
    "1210900964": (100, 101, 103),
    "1210913385": (6, 54, 55, 58),
    "1211437326": (5, 6, 7),
    "1212685850": (91, 92),
    "1212731673": (114, 116),
    "1212754192": (82, 83, 86),
    "1212749314": tuple(range(79, 97)),
    "1212944652": (157, 159),
    "1213098162": (1, 6, 7, 9),
    "1213177032": (9, 105, 106, 107, 108),
    "1216220927": (99, 100, 102),
    "1216282324": (136, 137, 138),
    "1216281417": (142, 143),
    "1216301205": (92, 93, 96),
    "1216417256": (82, 83),
    "1216576743": (2, 7, 9, 10),
    "1217649501": (51, 52, 54),
    "1217694603": (80, 81, 82),
    "1217717273": (7,),
    "1219391119": (123, 124, 127),
    "1219426026": (90, 91, 93),
    "1219448801": (128, 129, 130),
    "1219702367": (95, 96),
    "1219925343": (106, 107, 110),
    "1221057925": (7,),
    "1222869207": (120, 121, 124),
    "1222924572": (99, 100),
    "1222927545": (121,),
    "1223145398": (82, 84),
    "1223414665": (90, 91, 94),
    "1224602619": (55, 56, 57),
    "1224611848": (47,),
    "1224621469": (110,),
    "1225034758": (103, 157, 164),
    "1225046687": (98,),
    "1225062556": (171, 172),
    "1225063837": (97, 98, 99),
    "1225064857": (109, 110),
    "1225068291": (95, 96, 99),
    "1225117829": (72, 74),
    "1225179186": (7,),
}

# 个别扫描页的印章会破坏 Windows OCR 的中文行标签，但金额数字仍可跨倍率稳定识别。
# 这些条目来自逐页人工直读，并在准入时再次校验官方 PDF 页码、两期恒等式、已有总资产
# 以及至少两个渲染倍率的关键数字序列。
MANUAL_IMAGE_BALANCE_FACTS: dict[str, dict[str, Any]] = {
    "1208326876": {
        "page_number": 76,
        "unit": "千元",
        "current_assets": "1142444310",
        "current_liabilities": "875783807",
        "current_equity": "266660503",
        "prior_assets": "1056185927",
        "prior_liabilities": "810710931",
        "prior_equity": "245474996",
    },
}

MANUAL_IMAGE_ROW_FACTS: dict[
    str,
    dict[str, Any] | tuple[dict[str, Any], ...],
] = {
    "1202268049": {
        "metric_id": "PARENT_NET_PROFIT_YTD",
        "page_number": 4,
        "supporting_direct_text_page": 4,
        "source_section_label": "2.1.2 按中国企业会计准则编制的主要会计数据及财务指标",
        "unit": "百万元",
        "label": "归属于母公司股东的净（亏损）/利润",
        "current": "-13786",
        "prior": "6149",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1202789596": (
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 4,
            "source_section_label": "公司主要财务数据和股东变化",
            "unit": "千元",
            "label": "归属于本公司股东的扣除非经常性损益的净利润",
            "current": "870624",
            "prior": "960611",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 4,
            "source_section_label": "公司主要财务数据和股东变化",
            "unit": "千元",
            "label": "经营活动产生的现金流量净额",
            "current": "2418381",
            "prior": "1604415",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 17,
            "unit": "元",
            "label": "应收账款",
            "current": "1140846813.05",
            "prior": "1051642996.39",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 17,
            "unit": "元",
            "label": "存货",
            "current": "2453247187.65",
            "prior": "2543866145.35",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 18,
            "unit": "元",
            "label": "资产总计",
            "current": "25448875336.98",
            "prior": "15870577267.20",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 19,
            "unit": "元",
            "label": "负债合计",
            "current": "7752395186.37",
            "prior": "7186644118.29",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 21,
            "unit": "元",
            "label": "一、营业总收入",
            "current": "15543684867.01",
            "prior": "15070813896.71",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 22,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "1104564112.06",
            "prior": "1196687535.22",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 22,
            "unit": "元",
            "label": "归属于母公司所有者的净利润",
            "current": "1056422250.20",
            "prior": "960001444.60",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1203374405": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 8,
            "unit": "元",
            "label": "应收账款",
            "current": "206346665.78",
            "prior": "65416532.73",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 8,
            "unit": "元",
            "label": "存货",
            "current": "141582152.45",
            "prior": "147170244.62",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 10,
            "unit": "元",
            "label": "负债合计",
            "current": "917805091.59",
            "prior": "853270824.67",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 12,
            "unit": "百万元",
            "label": "营业利润",
            "current": "2580",
            "prior": "2381",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1202636649": {
        "metric_id": "TOTAL_LIABILITIES_END",
        "page_number": 28,
        "supporting_direct_text_page": 28,
        "unit": "元",
        "label": "负债合计",
        "current": "5526466486.70",
        "prior": "4072203945.04",
        "value_period_scope": "PERIOD_END",
    },
    "1202658580": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 59,
            "unit": "千元",
            "label": "应收账款",
            "current": "161600557",
            "prior": "131660359",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 59,
            "unit": "千元",
            "label": "存货",
            "current": "241779271",
            "prior": "246758108",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 59,
            "unit": "千元",
            "label": "负债合计",
            "current": "584164836",
            "prior": "574266577",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1203242141": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 108,
            "unit": "千元",
            "label": "应收账款",
            "current": "140532460",
            "prior": "131660359",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 108,
            "unit": "千元",
            "label": "存货",
            "current": "224804401",
            "prior": "246758108",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 108,
            "unit": "千元",
            "label": "负债合计",
            "current": "605350071",
            "prior": "574266577",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1204555388": {
        "metric_id": "OPERATING_PROFIT_YTD",
        "page_number": 125,
        "unit": "千元",
        "label": "三、营业利润",
        "current": "19247634",
        "prior": "17003565",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1204682861": (
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 108,
            "unit": "元",
            "label": "负债合计",
            "current": "17417178676.67",
            "prior": "14320872456.53",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 109,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "1950235812.69",
            "prior": "924009917.25",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1205334041": (
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 5,
            "source_section_label": "主要财务数据及指标",
            "unit": "百万元",
            "label": "归属于母公司股东的扣除非经常性损益后的净利润",
            "current": "39791",
            "prior": "26099",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 53,
            "unit": "百万元",
            "label": "存货",
            "current": "225573",
            "prior": "186683",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 53,
            "unit": "百万元",
            "label": "资产总计",
            "current": "1617304",
            "prior": "1595504",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 54,
            "unit": "百万元",
            "label": "负债合计",
            "current": "759993",
            "prior": "741434",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 57,
            "unit": "百万元",
            "label": "营业收入",
            "current": "1300252",
            "prior": "1165837",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 57,
            "unit": "百万元",
            "label": "营业利润",
            "current": "67940",
            "prior": "44917",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 57,
            "unit": "百万元",
            "label": "母公司股东的净利润",
            "current": "41600",
            "prior": "27092",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 59,
            "unit": "百万元",
            "label": "经营活动产生的现金流量净额",
            "current": "71620",
            "prior": "60847",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 78,
            "source_section_label": "附注7(b) 应收账款",
            "unit": "百万元",
            "label": "应收账款（减坏账准备后账面净额）",
            "current": "70912",
            "prior": "68494",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1205363576": {
        "metric_id": "OPERATING_PROFIT_YTD",
        "page_number": 71,
        "unit": "千元",
        "label": "三、营业利润",
        "current": "12467297",
        "prior": "10366408",
        "value_period_scope": "YEAR_TO_DATE",
        "manual_visual_verification": True,
    },
    "1205969212": (
        {
            "metric_id": "INVENTORY_END",
            "page_number": 133,
            "unit": "千元",
            "label": "存货",
            "current": "165241259",
            "prior": "127738498",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 134,
            "unit": "千元",
            "label": "负债合计",
            "current": "720532073",
            "prior": "674950262",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 137,
            "unit": "千元",
            "label": "三、营业利润",
            "current": "22696257",
            "prior": "19249829",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1212754192": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 82,
            "unit": "元",
            "label": "应收账款",
            "current": "7986787322.58",
            "prior": "7526238208.54",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 82,
            "unit": "元",
            "label": "存货",
            "current": "19062432842.88",
            "prior": "15609854069.57",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 83,
            "unit": "元",
            "label": "负债合计",
            "current": "67720696078.96",
            "prior": "64153537732.09",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 86,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "2658242870.02",
            "prior": "2160290654.85",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1212731673": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 114,
            "unit": "元",
            "label": "应收账款",
            "current": "58107770169.75",
            "prior": "52745905873.59",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 114,
            "unit": "元",
            "label": "存货",
            "current": "27104035290.12",
            "prior": "24088257693.63",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 116,
            "unit": "元",
            "label": "二、营业利润",
            "current": "8205486305.15",
            "prior": "7184388142.56",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1216282324": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 136,
            "unit": "元",
            "label": "应收账款",
            "current": "66759853950.45",
            "prior": "58107770169.75",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 136,
            "unit": "元",
            "label": "存货",
            "current": "34460216765.02",
            "prior": "27104035290.12",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 137,
            "unit": "元",
            "label": "负债合计",
            "current": "120132450441.52",
            "prior": "104369299024.61",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 138,
            "unit": "元",
            "label": "二、营业利润",
            "current": "9014650389.27",
            "prior": "8205486305.15",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1216417256": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 82,
            "unit": "元",
            "label": "应收账款",
            "current": "3527198489.52",
            "prior": "2659284095.13",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 82,
            "unit": "元",
            "label": "存货",
            "current": "2458679992.63",
            "prior": "2606847351.13",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 82,
            "unit": "元",
            "label": "负债合计",
            "current": "6274562705.66",
            "prior": "5316234034.23",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 83,
            "unit": "元",
            "label": "三、营业利润（亏损以“－”号填列）",
            "current": "1341977611.48",
            "prior": "1406886844.61",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1217694603": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 80,
            "unit": "元",
            "label": "应收账款",
            "current": "77414511678.20",
            "prior": "66759853950.45",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 80,
            "unit": "元",
            "label": "存货",
            "current": "33146468526.75",
            "prior": "34460216765.02",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 81,
            "unit": "元",
            "label": "负债合计",
            "current": "134375227951.51",
            "prior": "120132450441.52",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 82,
            "unit": "元",
            "label": "二、营业利润",
            "current": "4558084231.87",
            "prior": "5799204554.84",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1219448801": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 128,
            "unit": "元",
            "label": "应收账款",
            "current": "72933881117.17",
            "prior": "66759853950.45",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 128,
            "unit": "元",
            "label": "存货",
            "current": "36623393930.07",
            "prior": "34460216765.02",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 129,
            "unit": "元",
            "label": "负债合计",
            "current": "131646440801.87",
            "prior": "120132450441.52",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 130,
            "unit": "元",
            "label": "二、营业利润",
            "current": "7667234704.32",
            "prior": "9014650389.27",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1219702367": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 95,
            "unit": "元",
            "label": "应收账款",
            "current": "4338525630.25",
            "prior": "3527198489.52",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 95,
            "unit": "元",
            "label": "存货",
            "current": "2860764626.97",
            "prior": "2458679992.63",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 95,
            "unit": "元",
            "label": "负债合计",
            "current": "8067429921.55",
            "prior": "6279370110.80",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 96,
            "unit": "元",
            "label": "三、营业利润（亏损以“－”号填列）",
            "current": "1778084279.39",
            "prior": "1341977611.48",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1223145398": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 82,
            "unit": "元",
            "label": "应收账款",
            "current": "5758354119.70",
            "prior": "4338525630.25",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 82,
            "unit": "元",
            "label": "存货",
            "current": "3476970033.07",
            "prior": "2860764626.97",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 82,
            "unit": "元",
            "label": "负债合计",
            "current": "10818360027.95",
            "prior": "8067429921.55",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 84,
            "unit": "元",
            "label": "三、营业利润（亏损以“－”号填列）",
            "current": "2432233031.19",
            "prior": "1778084279.39",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1225117829": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 72,
            "unit": "元",
            "label": "应收账款",
            "current": "8183096459.67",
            "prior": "5758354119.70",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 72,
            "unit": "元",
            "label": "存货",
            "current": "4078944481.10",
            "prior": "3476970033.07",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 72,
            "unit": "元",
            "label": "负债合计",
            "current": "14039166962.60",
            "prior": "10818360027.95",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 74,
            "unit": "元",
            "label": "三、营业利润（亏损以“－”号填列）",
            "current": "3729437750.47",
            "prior": "2432233031.19",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1219391119": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 123,
            "unit": "元",
            "label": "应收账款",
            "current": "1132003814.45",
            "prior": "800256289.83",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 123,
            "unit": "元",
            "label": "存货",
            "current": "31430496020.23",
            "prior": "32254722426.64",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 124,
            "unit": "元",
            "label": "负债合计",
            "current": "101012356495.30",
            "prior": "102981802839.63",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 127,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "13287978156.38",
            "prior": "9889056473.98",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1222869207": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 120,
            "unit": "元",
            "label": "应收账款",
            "current": "647879043.30",
            "prior": "1132003814.45",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 120,
            "unit": "元",
            "label": "存货",
            "current": "29878326307.04",
            "prior": "31430496020.23",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 120,
            "unit": "元",
            "label": "资产总计",
            "current": "170236431691.82",
            "prior": "172974530702.61",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 121,
            "unit": "元",
            "label": "负债合计",
            "current": "84294195547.05",
            "prior": "101012356495.30",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 124,
            "unit": "元",
            "label": "三、营业利润（亏损以“－”号填列）",
            "current": "25266034420.32",
            "prior": "13287978156.38",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1224602619": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 55,
            "unit": "元",
            "label": "应收账款",
            "current": "86377845330.26",
            "prior": "79129205739.16",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 55,
            "unit": "元",
            "label": "存货",
            "current": "40899076963.47",
            "prior": "38943669737.08",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 56,
            "unit": "元",
            "label": "负债合计",
            "current": "149209151507.72",
            "prior": "137467066427.10",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 57,
            "unit": "元",
            "label": "二、营业利润",
            "current": "6833619624.55",
            "prior": "4830896881.37",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1212685850": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 91,
            "unit": "元",
            "label": "应收账款",
            "current": "4958727680",
            "prior": "4372904933",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 91,
            "unit": "元",
            "label": "存货",
            "current": "36976797713",
            "prior": "32687522034",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 92,
            "unit": "元",
            "label": "负债合计",
            "current": "83224717615",
            "prior": "74022247962",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1207047784": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 11,
            "unit": "千元",
            "label": "应收账款",
            "current": "20260192",
            "prior": "19390174",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 11,
            "unit": "千元",
            "label": "存货",
            "current": "24004608",
            "prior": "29645018",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 11,
            "unit": "千元",
            "label": "资产总计",
            "current": "285289199",
            "prior": "263701148",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 12,
            "unit": "千元",
            "label": "负债合计",
            "current": "180927263",
            "prior": "171246631",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 13,
            "unit": "千元",
            "label": "三、营业利润",
            "current": "7308785",
            "prior": "6476269",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1207684809": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 11,
            "unit": "千元",
            "label": "应收账款",
            "current": "20652085",
            "prior": "18663819",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 11,
            "unit": "千元",
            "label": "存货",
            "current": "24358547",
            "prior": "32443399",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 12,
            "unit": "千元",
            "label": "负债合计",
            "current": "190087022",
            "prior": "194459322",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 13,
            "unit": "千元",
            "label": "三、营业利润",
            "current": "5501245",
            "prior": "7683772",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1207688733": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 9,
            "unit": "百万元",
            "label": "应收账款",
            "current": "1769",
            "prior": "1717",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 9,
            "unit": "百万元",
            "label": "存货",
            "current": "2580",
            "prior": "2407",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 10,
            "unit": "百万元",
            "label": "负债合计",
            "current": "220613",
            "prior": "212539",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1208662817": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 11,
            "unit": "百万元",
            "label": "应收账款",
            "current": "1171",
            "prior": "1717",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 11,
            "unit": "百万元",
            "label": "存货",
            "current": "2350",
            "prior": "2407",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 11,
            "unit": "百万元",
            "label": "资产总计",
            "current": "283558",
            "prior": "282936",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 12,
            "unit": "百万元",
            "label": "负债合计",
            "current": "223741",
            "prior": "212539",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 14,
            "unit": "百万元",
            "label": "营业利润",
            "current": "-13071",
            "prior": "5524",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1208664350": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 13,
            "unit": "千元",
            "label": "应收账款",
            "current": "23076561",
            "prior": "18663819",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 13,
            "unit": "千元",
            "label": "存货",
            "current": "24130752",
            "prior": "32443399",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 13,
            "unit": "千元",
            "label": "资产总计",
            "current": "350443450",
            "prior": "301955419",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 14,
            "unit": "千元",
            "label": "负债合计",
            "current": "230314458",
            "prior": "194459322",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 15,
            "unit": "千元",
            "label": "三、营业利润",
            "current": "9500553",
            "prior": "7308785",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1209870320": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 11,
            "unit": "千元",
            "label": "应收账款",
            "current": "26388315",
            "prior": "22978363",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 11,
            "unit": "千元",
            "label": "存货",
            "current": "31496638",
            "prior": "31076529",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 12,
            "unit": "千元",
            "label": "负债合计",
            "current": "244805398",
            "prior": "236145503",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 13,
            "unit": "千元",
            "label": "三、营业利润",
            "current": "7413136",
            "prior": "5501245",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1216220927": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 99,
            "unit": "元",
            "label": "应收账款",
            "current": "4465150503",
            "prior": "4958727680",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 99,
            "unit": "元",
            "label": "存货",
            "current": "38061772570",
            "prior": "36976797713",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 100,
            "unit": "元",
            "label": "负债合计",
            "current": "85380380689",
            "prior": "83224717615",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 102,
            "unit": "元",
            "label": "营业利润",
            "current": "7646175807",
            "prior": "7429811486",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1217649501": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 51,
            "unit": "元",
            "label": "应收账款",
            "current": "4677762563",
            "prior": "4465150503",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 51,
            "unit": "元",
            "label": "存货",
            "current": "38932355382",
            "prior": "38061772570",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 52,
            "unit": "元",
            "label": "负债合计",
            "current": "125119938455",
            "prior": "85380380689",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 54,
            "unit": "元",
            "label": "营业利润",
            "current": "4335222386",
            "prior": "4718300083",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1219426026": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 90,
            "unit": "元",
            "label": "应收账款",
            "current": "3971108921",
            "prior": "4465150503",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 90,
            "unit": "元",
            "label": "存货",
            "current": "40538382252",
            "prior": "38061772570",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 91,
            "unit": "元",
            "label": "负债合计",
            "current": "91402239741",
            "prior": "85380380689",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 93,
            "unit": "元",
            "label": "营业利润",
            "current": "8418946985",
            "prior": "7646175807",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1222924572": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 99,
            "unit": "元",
            "label": "应收账款",
            "current": "6233407692",
            "prior": "3971108921",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 99,
            "unit": "元",
            "label": "存货",
            "current": "44853329771",
            "prior": "40538382252",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 100,
            "unit": "元",
            "label": "负债合计",
            "current": "105325778122",
            "prior": "91402239741",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1225034758": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 157,
            "unit": "元",
            "label": "应收账款：合计账面价值",
            "current": "6629834438",
            "prior": "6233407692",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 164,
            "unit": "元",
            "label": "存货：合计账面价值",
            "current": "68187864027",
            "prior": "44853329771",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1225062556": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 171,
            "unit": "元",
            "label": "应收账款",
            "current": "157728815136",
            "prior": "138018850273",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 171,
            "unit": "元",
            "label": "存货",
            "current": "114704691943",
            "prior": "102134242319",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 172,
            "unit": "元",
            "label": "负债合计",
            "current": "1551244760037",
            "prior": "1390457601488",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1225063837": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 97,
            "unit": "元",
            "label": "应收账款",
            "current": "82306400629.83",
            "prior": "79129205739.16",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 97,
            "unit": "元",
            "label": "存货",
            "current": "45358947550.04",
            "prior": "38943669737.08",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 98,
            "unit": "元",
            "label": "负债合计",
            "current": "143361473043.61",
            "prior": "137467066427.10",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 99,
            "unit": "元",
            "label": "二、营业利润",
            "current": "9942085632.77",
            "prior": "8140712946.17",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1225064857": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 109,
            "unit": "百万元",
            "label": "应收账款",
            "current": "2089",
            "prior": "1890",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 109,
            "unit": "百万元",
            "label": "存货",
            "current": "2695",
            "prior": "1680",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 110,
            "unit": "百万元",
            "label": "负债合计",
            "current": "252916",
            "prior": "235191",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1225179186": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 7,
            "unit": "千元",
            "label": "应收账款",
            "current": "24848541",
            "prior": "21670066",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 7,
            "unit": "千元",
            "label": "存货",
            "current": "51959114",
            "prior": "47017122",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1208627222": {
        "metric_id": "OPERATING_PROFIT_YTD",
        "page_number": 34,
        "supporting_direct_text_page": 34,
        "unit": "元",
        "label": "三、营业利润（亏损以“-”号填列）",
        "current": "3700869075.77",
        "prior": "480268626.81",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1209495189": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 130,
            "unit": "百万元",
            "label": "应收账款",
            "current": "1124",
            "prior": "1717",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 130,
            "unit": "百万元",
            "label": "存货",
            "current": "2054",
            "prior": "2407",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 131,
            "unit": "百万元",
            "label": "负债合计",
            "current": "225496",
            "prior": "212539",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1209464252": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 98,
            "unit": "元",
            "label": "应收账款",
            "current": "52745905873.59",
            "prior": "47339803537.21",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 98,
            "unit": "元",
            "label": "存货",
            "current": "24088257693.63",
            "prior": "24877356781.61",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 99,
            "unit": "元",
            "label": "负债合计",
            "current": "94444925708.98",
            "prior": "87640467566.32",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1207641522": {
        "metric_id": "ACCOUNTS_RECEIVABLE_END",
        "page_number": 135,
        "supporting_direct_text_page": 187,
        "unit": "元",
        "label": "应收账款",
        "current": "5890241538.78",
        "prior": "1378211622.25",
    },
    "1207586412": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 83,
            "unit": "元",
            "label": "应收账款",
            "current": "22115764161.40",
            "prior": "17349728013.80",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 83,
            "unit": "元",
            "label": "存货",
            "current": "14577009470.94",
            "prior": "13594101621.98",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 84,
            "unit": "元",
            "label": "负债合计",
            "current": "78273573882.29",
            "prior": "62004175341.01",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 87,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "3882125455.76",
            "prior": "2756455517.16",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1206869181": {
        "metric_id": "TOTAL_LIABILITIES_END",
        "page_number": 67,
        "unit": "千元",
        "label": "负债合计",
        "current": "782386880",
        "prior": "720532073",
        "value_period_scope": "PERIOD_END",
        "manual_visual_verification": True,
    },
    "1207423623": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 79,
            "unit": "百万元",
            "label": "应收账款",
            "current": "54865",
            "prior": "56993",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 79,
            "unit": "百万元",
            "label": "存货",
            "current": "192442",
            "prior": "184584",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 80,
            "unit": "百万元",
            "label": "负债合计",
            "current": "878166",
            "prior": "734649",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1207432718": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 89,
            "unit": "元",
            "label": "应收账款",
            "current": "4930446539",
            "prior": "5727719572",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 89,
            "unit": "元",
            "label": "存货",
            "current": "26923307427",
            "prior": "17259265461",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 90,
            "unit": "元",
            "label": "负债合计",
            "current": "75881313004",
            "prior": "50839136623",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 92,
            "unit": "元",
            "label": "营业利润",
            "current": "3197857832",
            "prior": "3265797504",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1207666500": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 12,
            "unit": "元",
            "label": "应收账款",
            "current": "395834467.73",
            "prior": "393038964.76",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 12,
            "unit": "元",
            "label": "存货",
            "current": "269907335.76",
            "prior": "263142354.59",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 14,
            "unit": "元",
            "label": "负债合计",
            "current": "2459750056.37",
            "prior": "1206469000.68",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1207691674": (
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 18,
            "unit": "百万元",
            "label": "营业（亏损）/利润",
            "current": "-9252",
            "prior": "29270",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 18,
            "unit": "百万元",
            "label": "归属于母公司股东的净（亏损）/利润",
            "current": "-16234",
            "prior": "10245",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1210900865": {
        "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
        "page_number": 7,
        "unit": "千元",
        "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
        "current": "-6920393",
        "prior": "-9596763",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1216576743": (
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 2,
            "source_section_label": "主要会计数据和财务指标（调整后）",
            "unit": "元",
            "label": "营业收入",
            "current": "630121089.65",
            "prior": "430712581.75",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 2,
            "source_section_label": "主要会计数据和财务指标（调整后）",
            "unit": "元",
            "label": "归属于上市公司股东的净利润",
            "current": "414065833.05",
            "prior": "273902632.86",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 2,
            "source_section_label": "主要会计数据和财务指标（调整后）",
            "unit": "元",
            "label": "归属于上市公司股东的扣除非经常性损益的净利润",
            "current": "387087745.92",
            "prior": "259338758.65",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 2,
            "source_section_label": "主要会计数据和财务指标（调整后）",
            "unit": "元",
            "label": "经营活动产生的现金流量净额",
            "current": "427008805.00",
            "prior": "240340450.10",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 2,
            "source_section_label": "主要会计数据和财务指标（调整后）",
            "unit": "元",
            "label": "总资产",
            "current": "6683188340.56",
            "prior": "6258547700.17",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 7,
            "unit": "元",
            "label": "应收账款",
            "current": "139097851.37",
            "prior": "127758045.77",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 7,
            "unit": "元",
            "label": "存货",
            "current": "37847042.51",
            "prior": "46715408.58",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 9,
            "unit": "元",
            "label": "负债合计",
            "current": "291506937.77",
            "prior": "312767027.94",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 10,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "485314712.31",
            "prior": "328053276.45",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1213098162": (
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 1,
            "source_section_label": "主要会计数据和财务指标",
            "unit": "元",
            "label": "营业收入",
            "current": "1102213029.27",
            "prior": "616714827.30",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 1,
            "source_section_label": "主要会计数据和财务指标",
            "unit": "元",
            "label": "归属于上市公司股东的净利润",
            "current": "164369327.24",
            "prior": "106064279.73",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 1,
            "source_section_label": "主要会计数据和财务指标",
            "unit": "元",
            "label": "归属于上市公司股东的扣除非经常性损益的净利润",
            "current": "141006734.64",
            "prior": "93256027.22",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 1,
            "source_section_label": "主要会计数据和财务指标",
            "unit": "元",
            "label": "经营活动产生的现金流量净额",
            "current": "57459668.08",
            "prior": "25664055.73",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 1,
            "source_section_label": "主要会计数据和财务指标",
            "unit": "元",
            "label": "总资产",
            "current": "8598676453.70",
            "prior": "6310960707.68",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 6,
            "unit": "元",
            "label": "应收账款",
            "current": "646732925.41",
            "prior": "482368604.79",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 6,
            "unit": "元",
            "label": "存货",
            "current": "1476713945.23",
            "prior": "1290839711.60",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 7,
            "unit": "元",
            "label": "负债合计",
            "current": "6071213971.77",
            "prior": "4070702178.04",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 9,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "185063305.63",
            "prior": "122548245.95",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1210899621": {
        "metric_id": "TOTAL_LIABILITIES_END",
        "page_number": 51,
        "supporting_direct_text_page": 51,
        "unit": "元",
        "label": "负债总计",
        "current": "93220641153",
        "prior": "74022247962",
        "value_period_scope": "PERIOD_END",
    },
    "1210913385": (
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 6,
            "unit": "百万元",
            "label": "营业收入",
            "current": "1261603",
            "prior": "1033064",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "unit": "百万元",
            "label": "归属于母公司股东的净利润/（亏损）",
            "current": "39153",
            "prior": "-23001",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "unit": "百万元",
            "label": "归属于母公司股东的扣除非经常性损益后的净利润/（亏损）",
            "current": "38420",
            "prior": "-24404",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 6,
            "unit": "百万元",
            "label": "经营活动产生的现金流量净额",
            "current": "47736",
            "prior": "40365",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 54,
            "unit": "百万元",
            "label": "应收账款",
            "current": "71843",
            "prior": "35587",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 54,
            "unit": "百万元",
            "label": "存货",
            "current": "199234",
            "prior": "151895",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 54,
            "unit": "百万元",
            "label": "资产总计",
            "current": "1852964",
            "prior": "1733805",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 55,
            "unit": "百万元",
            "label": "负债合计",
            "current": "945663",
            "prior": "849929",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 58,
            "unit": "百万元",
            "label": "营业利润/（亏损）",
            "current": "63862",
            "prior": "-27925",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1210900964": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 100,
            "unit": "千元",
            "label": "应收账款",
            "current": "184383313",
            "prior": "160441814",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 100,
            "unit": "千元",
            "label": "存货",
            "current": "700758214",
            "prior": "675125328",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 101,
            "unit": "千元",
            "label": "负债合计",
            "current": "1750203103",
            "prior": "1615078738",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 103,
            "unit": "千元",
            "label": "营业利润",
            "current": "51615934",
            "prior": "43259633",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1211437326": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 5,
            "unit": "千元",
            "label": "应收账款",
            "current": "25984593",
            "prior": "22978363",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 5,
            "unit": "千元",
            "label": "存货",
            "current": "33198146",
            "prior": "31076529",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 6,
            "unit": "千元",
            "label": "负债合计",
            "current": "248028302",
            "prior": "236145503",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 7,
            "unit": "千元",
            "label": "三、营业利润",
            "current": "27075496",
            "prior": "25875964",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1212944652": (
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 157,
            "unit": "千元",
            "label": "负债合计",
            "current": "1748546817",
            "prior": "1615231415",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 159,
            "unit": "千元",
            "label": "营业利润",
            "current": "100600900",
            "prior": "94463590",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1216301205": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 92,
            "unit": "元",
            "label": "应收账款",
            "current": "10092237182.40",
            "prior": "7986787322.58",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 92,
            "unit": "元",
            "label": "存货",
            "current": "18455259894.80",
            "prior": "19062432842.88",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 93,
            "unit": "元",
            "label": "负债合计",
            "current": "76640188940.23",
            "prior": "67720696078.96",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 96,
            "unit": "元",
            "label": "二、营业利润（亏损以“-”号填列）",
            "current": "3320871279.29",
            "prior": "2658242870.02",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1217717273": {
        "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
        "page_number": 7,
        "supporting_direct_text_page": 7,
        "unit": "千元",
        "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
        "current": "-4938159",
        "prior": "-19494182",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1221057925": {
        "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
        "page_number": 7,
        "supporting_direct_text_page": 7,
        "unit": "千元",
        "label": "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润",
        "current": "-3440209",
        "prior": "-4938159",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1223414665": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 90,
            "unit": "元",
            "label": "应收账款",
            "current": "12545314264.97",
            "prior": "10714105864.95",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 90,
            "unit": "元",
            "label": "存货",
            "current": "21685296057.95",
            "prior": "18136582872.93",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 91,
            "unit": "元",
            "label": "负债合计",
            "current": "98867036719.24",
            "prior": "79888498579.02",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 94,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "3887453074.86",
            "prior": "3976683797.57",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1224611848": {
        "metric_id": "OPERATING_PROFIT_YTD",
        "page_number": 47,
        "unit": "千元",
        "label": "二、营业亏损",
        "current": "-2923705",
        "prior": "-3477517",
        "value_period_scope": "YEAR_TO_DATE",
        "manual_visual_verification": True,
    },
    "1209481072": {
        "metric_id": "OPERATING_PROFIT_YTD",
        "page_number": 92,
        "unit": "元",
        "label": "营业利润",
        "current": "3318646709",
        "prior": "3197857832",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1209477137": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 121,
            "unit": "元",
            "label": "应收账款",
            "current": "1144340790.93",
            "prior": "286138349.80",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 121,
            "unit": "元",
            "label": "存货",
            "current": "250455974513.76",
            "prior": "185344146255.40",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 122,
            "unit": "元",
            "label": "负债合计",
            "current": "346212728756.87",
            "prior": "284626613955.23",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1209490194": {
        "metric_id": "OPERATING_PROFIT_YTD",
        "page_number": 86,
        "unit": "千元",
        "label": "二、营业（亏损）利润",
        "current": "-18499903",
        "prior": "9178331",
        "value_period_scope": "YEAR_TO_DATE",
    },
    "1209726071": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 144,
            "unit": "元",
            "label": "应收账款",
            "current": "3879744130.04",
            "prior": "5890241538.78",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 144,
            "unit": "元",
            "label": "存货",
            "current": "9650858867.17",
            "prior": "9153238548.05",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 145,
            "unit": "元",
            "label": "负债合计",
            "current": "61968119374.71",
            "prior": "56211005916.20",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1222927545": {
        "metric_id": "INVENTORY_END",
        "page_number": 121,
        "unit": "元",
        "label": "存货",
        "current": "38943669737.08",
        "prior": "36623393930.07",
        "value_period_scope": "PERIOD_END",
    },
    "1224621469": {
        "metric_id": "INVENTORY_END",
        "page_number": 110,
        "unit": "元",
        "label": "存货",
        "current": "113069072846",
        "prior": "102134242319",
        "value_period_scope": "PERIOD_END",
    },
    "1225046687": {
        "metric_id": "INVENTORY_END",
        "page_number": 98,
        "unit": "元",
        "label": "流动资产：存货",
        "current": "40600637422.42",
        "prior": "29878326307.04",
        "value_period_scope": "PERIOD_END",
    },
    "1225068291": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 95,
            "unit": "元",
            "label": "应收账款",
            "current": "15193794901.36",
            "prior": "12545314264.97",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 95,
            "unit": "元",
            "label": "存货",
            "current": "26171153034.44",
            "prior": "21685296057.95",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 96,
            "unit": "元",
            "label": "负债合计",
            "current": "114505939883.61",
            "prior": "98867036719.24",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 99,
            "unit": "元",
            "label": "三、营业利润（亏损以“-”号填列）",
            "current": "4778268464.68",
            "prior": "3887453074.86",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1219925343": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 106,
            "unit": "元",
            "label": "应收账款",
            "current": "180380469",
            "prior": "105252340",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 106,
            "unit": "元",
            "label": "存货",
            "current": "202063986",
            "prior": "174787526",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 107,
            "unit": "元",
            "label": "负债合计",
            "current": "28487992833",
            "prior": "29726790820",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 110,
            "unit": "元",
            "label": "二、营业利润/(亏损)",
            "current": "2616271913",
            "prior": "-3416652099",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1208637839": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 27,
            "unit": "千元",
            "label": "应收账款",
            "current": "14472119",
            "prior": "19778280",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 27,
            "unit": "千元",
            "label": "存货",
            "current": "39353293",
            "prior": "27688508",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 27,
            "unit": "千元",
            "label": "资产总计",
            "current": "165268109",
            "prior": "141202135",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 28,
            "unit": "千元",
            "label": "负债合计",
            "current": "120511299",
            "prior": "103247837",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 29,
            "unit": "千元",
            "label": "二、营业利润",
            "current": "3939014",
            "prior": "5883016",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1213177032": (
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 9,
            "unit": "元",
            "label": "归属于上市公司股东的扣除非经常性损益的净利润",
            "current": "9437240976",
            "prior": "2933248153",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 105,
            "unit": "千元",
            "label": "应收账款",
            "current": "18238782",
            "prior": "12557614",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 105,
            "unit": "千元",
            "label": "存货",
            "current": "14083357",
            "prior": "8834958",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 105,
            "unit": "千元",
            "label": "资产总计",
            "current": "308733132",
            "prior": "257908279",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 106,
            "unit": "千元",
            "label": "负债合计",
            "current": "189087840",
            "prior": "167851212",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 107,
            "unit": "千元",
            "label": "其中：营业收入",
            "current": "163540560",
            "prior": "76677238",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 107,
            "unit": "千元",
            "label": "二、营业利润",
            "current": "17352567",
            "prior": "5359904",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 107,
            "unit": "千元",
            "label": "归属于母公司所有者的净利润",
            "current": "10057444",
            "prior": "4388159",
            "value_period_scope": "YEAR_TO_DATE",
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 108,
            "unit": "千元",
            "label": "经营活动产生的现金流量净额",
            "current": "32878453",
            "prior": "16698283",
            "value_period_scope": "YEAR_TO_DATE",
        },
    ),
    "1216281417": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 142,
            "unit": "百万元",
            "label": "应收账款",
            "current": "754",
            "prior": "974",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 142,
            "unit": "百万元",
            "label": "存货",
            "current": "1626",
            "prior": "1799",
            "value_period_scope": "PERIOD_END",
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 143,
            "unit": "百万元",
            "label": "负债合计",
            "current": "255641",
            "prior": "231638",
            "value_period_scope": "PERIOD_END",
        },
    ),
    "1202251607": (
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "营业收入",
            "current": "2804527234.25",
            "prior": "3136833212.74",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 16,
            "unit": "元",
            "label": "三、营业利润（亏损以‘－’号填列）",
            "current": "124369630.79",
            "prior": "142786229.77",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "归属于上市公司股东的净利润",
            "current": "73085510.15",
            "prior": "101577771.53",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "归属于上市公司股东的扣除非经常性损益的净利润",
            "current": "63679383.91",
            "prior": "98585940.86",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "经营活动产生的现金流量净额",
            "current": "25316727.93",
            "prior": "35971881.24",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 11,
            "unit": "元",
            "label": "应收账款",
            "current": "1042551067.34",
            "prior": "710204663.76",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 11,
            "unit": "元",
            "label": "存货",
            "current": "3798320109.60",
            "prior": "3609563367.12",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 12,
            "unit": "元",
            "label": "资产总计",
            "current": "34687794628.88",
            "prior": "34789179528.87",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 13,
            "unit": "元",
            "label": "负债合计",
            "current": "9076740844.49",
            "prior": "9274217015.00",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1202791853": (
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "营业收入",
            "current": "9915313881.36",
            "prior": "10387007503.53",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 18,
            "unit": "元",
            "label": "三、营业利润（亏损以‘－’号填列）",
            "current": "614402828.09",
            "prior": "802852568.31",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "归属于上市公司股东的净利润",
            "current": "461561082.73",
            "prior": "541442232.39",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "归属于上市公司股东的扣除非经常性损益的净利润",
            "current": "430825310.47",
            "prior": "547443781.09",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 3,
            "unit": "元",
            "label": "经营活动产生的现金流量净额",
            "current": "457136009.22",
            "prior": "542129958.27",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 11,
            "unit": "元",
            "label": "应收账款",
            "current": "1353028122.84",
            "prior": "710204663.76",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 11,
            "unit": "元",
            "label": "存货",
            "current": "3854693897.52",
            "prior": "3609563367.12",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 12,
            "unit": "元",
            "label": "资产总计",
            "current": "34734214309.60",
            "prior": "34789179528.87",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 13,
            "unit": "元",
            "label": "负债合计",
            "current": "8932937619.85",
            "prior": "9274217015.00",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1202795561": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 12,
            "unit": "百万元",
            "label": "应收账款",
            "current": "2651",
            "prior": "2867",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 12,
            "unit": "百万元",
            "label": "存货",
            "current": "2276",
            "prior": "2056",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 13,
            "unit": "百万元",
            "label": "负债合计",
            "current": "151437",
            "prior": "158058",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1203416615": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 11,
            "unit": "百万元",
            "label": "应收账款",
            "current": "2245",
            "prior": "2630",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 11,
            "unit": "百万元",
            "label": "存货",
            "current": "2285",
            "prior": "2248",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 12,
            "unit": "百万元",
            "label": "负债合计",
            "current": "156697",
            "prior": "159955",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1203848634": (
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 6,
            "unit": "元",
            "label": "营业收入",
            "current": "1369252218.01",
            "prior": "1164425414.31",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 47,
            "unit": "元",
            "label": "三、营业利润（亏损以‘－’号填列）",
            "current": "265108910.32",
            "prior": "210573547.26",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "unit": "元",
            "label": "归属于上市公司股东的净利润",
            "current": "233067650.65",
            "prior": "178136013.50",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "CORE_PARENT_NET_PROFIT_YTD",
            "page_number": 6,
            "unit": "元",
            "label": "归属于上市公司股东的扣除非经常性损益的净利润",
            "current": "231360197.74",
            "prior": "179298133.93",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_CASH_FLOW_YTD",
            "page_number": 6,
            "unit": "元",
            "label": "经营活动产生的现金流量净额",
            "current": "311557921.86",
            "prior": "129653602.17",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 41,
            "unit": "元",
            "label": "应收账款",
            "current": "202519139.52",
            "prior": "65416532.73",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 41,
            "unit": "元",
            "label": "存货",
            "current": "156094802.77",
            "prior": "147170244.62",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_ASSETS_END",
            "page_number": 42,
            "unit": "元",
            "label": "资产总计",
            "current": "4199359140.85",
            "prior": "3758156522.19",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 43,
            "unit": "元",
            "label": "负债合计",
            "current": "1129689026.41",
            "prior": "853270824.67",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1204081143": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 12,
            "unit": "百万元",
            "label": "应收账款",
            "current": "2680",
            "prior": "2630",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 12,
            "unit": "百万元",
            "label": "存货",
            "current": "2400",
            "prior": "2248",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 13,
            "unit": "百万元",
            "label": "负债合计",
            "current": "165575",
            "prior": "159955",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 15,
            "unit": "百万元",
            "label": "二、营业利润",
            "current": "9943",
            "prior": "5509",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1204803034": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 9,
            "unit": "百万元",
            "label": "应收账款",
            "current": "1395",
            "prior": "2124",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 9,
            "unit": "百万元",
            "label": "存货",
            "current": "2186",
            "prior": "2185",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 10,
            "unit": "百万元",
            "label": "负债合计",
            "current": "168502",
            "prior": "170946",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1207036153": (
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "page_number": 12,
            "unit": "百万元",
            "label": "营业收入",
            "current": "93400",
            "prior": "87878",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "OPERATING_PROFIT_YTD",
            "page_number": 12,
            "unit": "百万元",
            "label": "营业利润",
            "current": "5524",
            "prior": "5759",
            "value_period_scope": "YEAR_TO_DATE",
            "manual_visual_verification": True,
        },
    ),
    "1205968967": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 165,
            "unit": "百万元",
            "label": "应收账款",
            "current": "1432",
            "prior": "2124",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 112,
            "unit": "百万元",
            "label": "存货",
            "current": "1950",
            "prior": "2185",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 113,
            "unit": "百万元",
            "label": "负债合计",
            "current": "177413",
            "prior": "170946",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
    "1207441457": (
        {
            "metric_id": "ACCOUNTS_RECEIVABLE_END",
            "page_number": 119,
            "unit": "百万元",
            "label": "应收账款",
            "current": "1717",
            "prior": "1432",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "INVENTORY_END",
            "page_number": 119,
            "unit": "百万元",
            "label": "存货",
            "current": "2407",
            "prior": "1950",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
        {
            "metric_id": "TOTAL_LIABILITIES_END",
            "page_number": 120,
            "unit": "百万元",
            "label": "负债合计",
            "current": "212539",
            "prior": "177413",
            "value_period_scope": "PERIOD_END",
            "manual_visual_verification": True,
        },
    ),
}

MANUAL_IMAGE_RECEIVABLE_RECONCILIATION_FACTS: dict[str, dict[str, Any]] = {
    "1205361911": {
        "metric_id": "ACCOUNTS_RECEIVABLE_END",
        "page_number": 112,
        "unit": "千元",
        "label": "应收账款；减：坏账准备",
        "current_gross": "4791455",
        "current_allowance": "185634",
        "current_bills": "363",
        "current_combined": "4606184",
        "prior_gross": "3674827",
        "prior_allowance": "184400",
        "prior_bills": "348",
        "prior_combined": "3490775",
    },
}

MANUAL_EXCLUSIVE_OCR_PAGE_ANNOUNCEMENT_IDS = frozenset(
    set(MANUAL_IMAGE_ROW_FACTS) | set(MANUAL_IMAGE_RECEIVABLE_RECONCILIATION_FACTS)
)

_FINANCIAL_REPORT_TOC_LINE_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百\d]+[章节])?"
    r"财务报告(?:[（(][^）)]*[）)])?[.…·。]*(?P<page>\d{1,3})$"
)


def _financial_report_toc_start_pages(
    page_texts: Sequence[str],
) -> tuple[int, ...]:
    """从报告前部目录提取 0 起始的财务报告起始页。"""

    starts: set[int] = set()
    for page_text in page_texts[:12]:
        for raw_line in str(page_text).splitlines():
            compact = re.sub(r"\s+", "", raw_line)
            match = _FINANCIAL_REPORT_TOC_LINE_PATTERN.fullmatch(compact)
            if match is None:
                continue
            page_number = int(match.group("page"))
            if 1 <= page_number <= len(page_texts):
                starts.add(page_number - 1)
    return tuple(sorted(starts))
TARGET_METRICS = {
    "OPERATING_REVENUE_YTD",
    "OPERATING_PROFIT_YTD",
    "PARENT_NET_PROFIT_YTD",
    "CORE_PARENT_NET_PROFIT_YTD",
    "ACCOUNTS_RECEIVABLE_END",
    "INVENTORY_END",
    "TOTAL_LIABILITIES_END",
}
BALANCE_METRICS = {
    "ACCOUNTS_RECEIVABLE_END",
    "INVENTORY_END",
    "TOTAL_ASSETS_END",
    "TOTAL_LIABILITIES_END",
    "TOTAL_EQUITY_END",
    "TOTAL_LIABILITIES_EQUITY_END",
}
INCOME_METRICS = {
    "OPERATING_REVENUE_YTD",
    "OPERATING_PROFIT_YTD",
    "PARENT_NET_PROFIT_YTD",
    "CORE_PARENT_NET_PROFIT_YTD",
}
AUXILIARY_INCOME_METRICS = {
    "NONOPERATING_INCOME_YTD",
    "NONOPERATING_EXPENSE_YTD",
    "PROFIT_TOTAL_YTD",
}
OCR_PATTERNS: dict[str, re.Pattern[str]] = {
    "OPERATING_REVENUE_YTD": re.compile(r"(?<!总)营业收入"),
    "OPERATING_PROFIT_YTD": re.compile(r"(?<!非)营业利润"),
    "PARENT_NET_PROFIT_YTD": re.compile(
        r"(?:归属|属于)于?母公司(?:所有者|股东)的净利润|归属于本公司股东的净利润"
    ),
    "ACCOUNTS_RECEIVABLE_END": re.compile(
        r"(?<!应收票据及)(?<!应收票据和)(?<!其他)应收账款"
    ),
    "INVENTORY_END": re.compile(r"(?<!周转)存货"),
    "TOTAL_ASSETS_END": re.compile(r"(?<!流动)(?<!非流动)资产总计"),
    "TOTAL_LIABILITIES_END": re.compile(
        r"(?<!流动)(?<!非流动)负[债偾](?:合计|总计)"
    ),
    "TOTAL_EQUITY_END": re.compile(
        r"(?<!负债和)(?<!负债及)(?<!母公司)(?<!少数)"
        r"(?:股东|所有者)权益(?:合计|总计)"
    ),
    "TOTAL_LIABILITIES_EQUITY_END": re.compile(
        r"负[债偾](?:和|及)(?:股东|所有者)权益(?:总计|合计)"
    ),
    "PROFIT_TOTAL_YTD": re.compile(r"(?<!营业)利润总额"),
}

NOTE_TABLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "NONOPERATING_INCOME_YTD": re.compile(r"营业外收入"),
    "NONOPERATING_EXPENSE_YTD": re.compile(r"营业外支出"),
}

_TARGETED_NUMBER_TRANSLATION = str.maketrans(
    {
        ":": ",",
        "：": ",",
        "》": ",",
        "〉": ",",
        "﹥": ",",
    }
)


@dataclass(frozen=True)
class OcrObservation:
    metric_id: str
    page_index: int
    render_scale: float
    amount_index: int
    raw_value: str
    value: Decimal
    row_text: str
    pixel_width: int
    pixel_height: int
    accounting_identity_verified: bool = False


@dataclass(frozen=True)
class ParsedOcrScale:
    observations: tuple[OcrObservation, ...]
    row_presence: frozenset[tuple[str, int]]


def _row_numeric_values(
    row: dict[str, Any],
) -> list[tuple[str, Decimal]]:
    numeric: list[tuple[str, Decimal]] = []
    for segment in row["segments"]:
        raw_value = str(segment["text"])
        value = parser._v1_6._normalize_ocr_number(
            raw_value.translate(_TARGETED_NUMBER_TRANSLATION)
        )
        if value is not None:
            numeric.append((raw_value, value))
    return numeric


def _note_table_observations(
    record: dict[str, Any],
) -> tuple[list[OcrObservation], set[tuple[str, int]]]:
    rows = parser._v1_6._v1_3._ocr_rows(record)
    observations: list[OcrObservation] = []
    presence: set[tuple[str, int]] = set()
    for metric_id, heading_pattern in NOTE_TABLE_PATTERNS.items():
        heading_indices = [
            index
            for index, row in enumerate(rows)
            if heading_pattern.search(str(row["compact"])) is not None
            and re.search(r"(?:^|[^0-9])\d{1,3}[、.]?", str(row["compact"]))
        ]
        for heading_index in heading_indices:
            header_index = next(
                (
                    index
                    for index in range(
                        heading_index + 1,
                        min(len(rows), heading_index + 6),
                    )
                    if (
                        "发生额" in str(rows[index]["compact"])
                        and (
                            "本年" in str(rows[index]["compact"])
                            or "本期" in str(rows[index]["compact"])
                        )
                    )
                ),
                None,
            )
            if header_index is None:
                continue
            component_values: list[Decimal] = []
            for row_index in range(
                header_index + 1,
                min(len(rows), header_index + 24),
            ):
                compact = str(rows[row_index]["compact"])
                if re.match(r"^\d{1,3}[、.]", compact):
                    break
                numeric = _row_numeric_values(rows[row_index])
                if len(numeric) < 3:
                    continue
                current_raw, current_value = numeric[0]
                duplicate_raw, duplicate_value = numeric[-1]
                if current_value != duplicate_value:
                    continue
                if (
                    len(component_values) >= 2
                    and sum(component_values, Decimal("0")) == current_value
                ):
                    selected_row = rows[row_index]
                    page_index = int(selected_row["page_index"])
                    observations.append(
                        OcrObservation(
                            metric_id=metric_id,
                            page_index=page_index,
                            render_scale=float(selected_row["render_scale"]),
                            amount_index=0,
                            raw_value=current_raw,
                            value=current_value,
                            row_text=compact[:1000],
                            pixel_width=int(selected_row["pixel_width"]),
                            pixel_height=int(selected_row["pixel_height"]),
                            accounting_identity_verified=True,
                        )
                    )
                    presence.add((metric_id, page_index))
                    break
                component_values.append(current_value)
    return observations, presence


def _metric_cny(result: dict[str, Any], metric_id: str) -> Decimal | None:
    return parser._v1_6._metric_cny(result, metric_id)


def _same_display(left: Decimal, right: Decimal, unit: str) -> bool:
    return parser._v1_4._same_display_identity(left, right, unit)


def _parse_scale(records: Sequence[dict[str, Any]]) -> ParsedOcrScale:
    observations: list[OcrObservation] = []
    row_presence: set[tuple[str, int]] = set()
    for record in records:
        note_observations, note_presence = _note_table_observations(record)
        observations.extend(note_observations)
        row_presence.update(note_presence)
        for row in parser._v1_6._v1_3._ocr_rows(record):
            compact = str(row["compact"])
            for metric_id, pattern in OCR_PATTERNS.items():
                if pattern.search(compact) is None:
                    continue
                if metric_id == "ACCOUNTS_RECEIVABLE_END" and any(
                    marker in compact
                    for marker in (
                        "应收票据及应收账款",
                        "应收票据和应收账款",
                        "应收账款及合同资产",
                        "应收账款和合同资产",
                    )
                ):
                    continue
                if metric_id == "TOTAL_EQUITY_END" and re.search(
                    r"负[债偾](?:和|及)(?:股东|所有者)权益",
                    compact,
                ):
                    continue
                page_index = int(row["page_index"])
                row_presence.add((metric_id, page_index))
                numeric = _row_numeric_values(row)
                while (
                    len(numeric) >= 3
                    and numeric[0][1] == numeric[0][1].to_integral_value()
                    and abs(numeric[0][1]) <= 999
                ):
                    numeric = numeric[1:]
                if not numeric:
                    continue
                amount_indices = [0]
                if metric_id in INCOME_METRICS and len(numeric) >= 4:
                    amount_indices.append(2)
                for amount_index in amount_indices:
                    if amount_index >= len(numeric):
                        continue
                    raw_value, value = numeric[amount_index]
                    observations.append(
                        OcrObservation(
                            metric_id=metric_id,
                            page_index=page_index,
                            render_scale=float(row["render_scale"]),
                            amount_index=amount_index,
                            raw_value=raw_value,
                            value=value,
                            row_text=compact[:1000],
                            pixel_width=int(row["pixel_width"]),
                            pixel_height=int(row["pixel_height"]),
                        )
                    )
    observations.sort(
        key=lambda item: (
            item.metric_id,
            item.page_index,
            item.amount_index,
            item.value,
        )
    )
    return ParsedOcrScale(
        observations=tuple(observations),
        row_presence=frozenset(row_presence),
    )


def _pairs(
    parsed: dict[float, ParsedOcrScale],
    metric_id: str,
) -> list[tuple[OcrObservation, OcrObservation]]:
    result: list[tuple[OcrObservation, OcrObservation]] = []
    scales = tuple(scale for scale in DIRECT_OCR_RENDER_SCALES if scale in parsed)
    for left_index, left_scale in enumerate(scales):
        left_values = [
            observation
            for observation in parsed[left_scale].observations
            if observation.metric_id == metric_id
            and (
                observation.accounting_identity_verified
                or _valid_exact_display(observation.raw_value)
            )
        ]
        for right_scale in scales[left_index + 1 :]:
            right_values = [
                observation
                for observation in parsed[right_scale].observations
                if observation.metric_id == metric_id
                and (
                    observation.accounting_identity_verified
                    or _valid_exact_display(observation.raw_value)
                )
            ]
            for left in left_values:
                for right in right_values:
                    if (
                        left.page_index == right.page_index
                        and left.amount_index == right.amount_index
                        and left.value == right.value
                    ):
                        result.append((left, right))
    return result


def _translated_number_text(raw_value: str) -> str:
    return (
        re.sub(r"\s+", "", raw_value)
        .translate(_TARGETED_NUMBER_TRANSLATION)
        .translate(parser._v1_6._v1_3.OCR_NUMBER_TRANSLATION)
    )


def _valid_exact_display(raw_value: str) -> bool:
    text = _translated_number_text(raw_value).strip("()")
    if "," not in text:
        return True
    unsigned = text.lstrip("-")
    integer = unsigned.split(".", 1)[0]
    return re.fullmatch(r"\d{1,3}(?:,\d{3})+", integer) is not None


def _canonical_digits(raw_value: str) -> str:
    digits = "".join(re.findall(r"\d", _translated_number_text(raw_value)))
    return digits.lstrip("0") or "0"


def _has_explicit_two_decimal_digits(raw_value: str) -> bool:
    text = _translated_number_text(raw_value).strip("()")
    return re.search(r"\.\d{2}$", text) is not None


def _digit_reconciled_pair(
    parsed: dict[float, ParsedOcrScale],
    metric_id: str,
    *,
    pages: set[int],
    amount_index: int,
) -> tuple[OcrObservation, OcrObservation] | None:
    observations = [
        observation
        for scale in DIRECT_OCR_RENDER_SCALES
        if scale in parsed
        for observation in parsed[scale].observations
        if observation.metric_id == metric_id
        and observation.page_index in pages
        and observation.amount_index == amount_index
    ]
    candidates: dict[Decimal, tuple[OcrObservation, OcrObservation]] = {}
    for explicit in observations:
        if not _has_explicit_two_decimal_digits(explicit.raw_value):
            continue
        digits = _canonical_digits(explicit.raw_value)
        for peer in observations:
            if (
                peer.render_scale == explicit.render_scale
                or peer.page_index != explicit.page_index
                or _canonical_digits(peer.raw_value) != digits
            ):
                continue
            selected = candidates.get(explicit.value)
            if selected is None or explicit.page_index < selected[1].page_index:
                candidates[explicit.value] = (peer, explicit)
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _pair_key(
    pair: tuple[OcrObservation, OcrObservation],
) -> tuple[int, int, Decimal]:
    return pair[1].page_index, pair[1].amount_index, pair[1].value


def _pairs_on_pages(
    pairs: Iterable[tuple[OcrObservation, OcrObservation]],
    pages: set[int],
    *,
    amount_index: int | None = None,
) -> list[tuple[OcrObservation, OcrObservation]]:
    return [
        pair
        for pair in pairs
        if pair[1].page_index in pages
        and (amount_index is None or pair[1].amount_index == amount_index)
    ]


def _unique_pair_by_value(
    pairs: Sequence[tuple[OcrObservation, OcrObservation]],
) -> tuple[OcrObservation, OcrObservation] | None:
    unique: dict[Decimal, tuple[OcrObservation, OcrObservation]] = {}
    for pair in pairs:
        unique[pair[1].value] = pair
    return next(iter(unique.values())) if len(unique) == 1 else None


def _ocr_evidence(
    metric_id: str,
    pair: tuple[OcrObservation, OcrObservation],
    *,
    unit: str,
    validation: dict[str, Any],
    source_context: dict[str, Any],
) -> parser._base.MetricEvidence:
    first, second = pair
    value_period_scope = "PERIOD_END" if metric_id.endswith("_END") else "YEAR_TO_DATE"
    section = (
        "TARGETED_RECONCILED_CONSOLIDATED_BALANCE_SHEET"
        if value_period_scope == "PERIOD_END"
        else "TARGETED_RECONCILED_CONSOLIDATED_INCOME_STATEMENT"
    )
    locator = {
        "page": second.page_index + 1,
        "section": section,
        "parser_version": DIRECT_OCR_VERSION,
        "ocr_engine": parser._v1_6._v1_3.OCR_ENGINE_NAME,
        "ocr_language_tag": parser._v1_6._v1_3.OCR_LANGUAGE_TAG,
        "render_scales": [first.render_scale, second.render_scale],
        "selected_amount_index": second.amount_index,
        "raw_value_by_scale": {
            str(first.render_scale): first.raw_value,
            str(second.render_scale): second.raw_value,
        },
        "normalized_value_by_scale": {
            str(first.render_scale): str(first.value),
            str(second.render_scale): str(second.value),
        },
        "row_text_by_scale": {
            str(first.render_scale): first.row_text,
            str(second.render_scale): second.row_text,
        },
        "pixel_dimensions_by_scale": {
            str(first.render_scale): [first.pixel_width, first.pixel_height],
            str(second.render_scale): [second.pixel_width, second.pixel_height],
        },
        "validation": validation,
        "source_context": source_context,
    }
    return parser._base.MetricEvidence(
        metric_id=metric_id,
        metric_value_cny=float(second.value * parser._base.UNIT_MULTIPLIERS[unit]),
        statement_scope="CONSOLIDATED_ONLY",
        value_period_scope=value_period_scope,
        source_page=second.page_index + 1,
        source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
        source_label=second.row_text.split(second.raw_value, 1)[0],
        source_raw_value=second.raw_value,
        source_unit=unit,
        source_unit_multiplier=float(parser._base.UNIT_MULTIPLIERS[unit]),
        source_method=validation.get(
            "source_method",
            "WINDOWS_MEDIA_OCR_TARGETED_DIRECT_PDF_DUAL_RENDER_SCALE_"
            "EXACT_VALUE_ACCOUNTING_IDENTITY_RECONCILED",
        ),
        verification_status=DIRECT_OCR_VERIFICATION_STATUS,
    )


def _asset_anchor_candidates(
    result: dict[str, Any],
    parsed: dict[float, ParsedOcrScale],
) -> list[dict[str, Any]]:
    expected_assets = _metric_cny(result, "TOTAL_ASSETS_END")
    if expected_assets is None:
        return []
    expected_unit = direct._metric_source_unit(result, "TOTAL_ASSETS_END")
    candidates: list[dict[str, Any]] = []
    for pair in _pairs(parsed, "TOTAL_ASSETS_END"):
        if pair[1].amount_index != 0:
            continue
        for unit in parser._base.UNIT_MULTIPLIERS:
            observed_cny = pair[1].value * parser._base.UNIT_MULTIPLIERS[unit]
            if _same_display(observed_cny, expected_assets, expected_unit or unit):
                candidates.append({"pair": pair, "unit": unit})
    return candidates


def _single_scale_equity_identity(
    parsed: dict[float, ParsedOcrScale],
    *,
    pages: set[int],
    expected_value: Decimal,
) -> dict[str, Any] | None:
    scales = tuple(scale for scale in DIRECT_OCR_RENDER_SCALES if scale in parsed)
    candidates: list[dict[str, Any]] = []
    for scale in scales:
        for observation in parsed[scale].observations:
            if (
                observation.metric_id != "TOTAL_EQUITY_END"
                or observation.page_index not in pages
                or observation.amount_index != 0
                or abs(observation.value - expected_value) > Decimal("1")
            ):
                continue
            peer_scales = [
                peer_scale
                for peer_scale in scales
                if peer_scale != scale
                and (
                    "TOTAL_EQUITY_END",
                    observation.page_index,
                ) in parsed[peer_scale].row_presence
            ]
            if not peer_scales:
                continue
            candidates.append(
                {
                    "observation": observation,
                    "peer_scale": min(peer_scales),
                    "expected_value": expected_value,
                }
            )
    unique = {
        (candidate["observation"].page_index, candidate["observation"].value): candidate
        for candidate in candidates
    }
    return next(iter(unique.values())) if len(unique) == 1 else None


def _liability_pairs_with_digit_reconciliation(
    parsed: dict[float, ParsedOcrScale],
    *,
    pages: set[int],
) -> list[tuple[tuple[OcrObservation, OcrObservation], bool]]:
    exact_pairs = _pairs_on_pages(
        _pairs(parsed, "TOTAL_LIABILITIES_END"),
        pages,
        amount_index=0,
    )
    candidates = [(pair, False) for pair in exact_pairs]
    reconciled = _digit_reconciled_pair(
        parsed,
        "TOTAL_LIABILITIES_END",
        pages=pages,
        amount_index=0,
    )
    if reconciled is not None and not any(
        _pair_key(pair) == _pair_key(reconciled) for pair, _ in candidates
    ):
        candidates.append((reconciled, True))
    return candidates


def _add_balance_metrics(
    result: dict[str, Any],
    parsed: dict[float, ParsedOcrScale],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    needed = {
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "TOTAL_LIABILITIES_END",
    }.intersection(result.get("missing_metrics") or [])
    if not needed:
        return {
            "status": "NOT_NEEDED_NO_TARGETED_OCR_BALANCE_METRIC_MISSING",
            "admitted_metric_ids": [],
        }
    structural: list[dict[str, Any]] = []
    equity_pairs = _pairs(parsed, "TOTAL_EQUITY_END")
    total_pairs = _pairs(parsed, "TOTAL_LIABILITIES_EQUITY_END")
    asset_row_pages = {
        observation.page_index
        for scale in parsed.values()
        for observation in scale.observations
        if observation.metric_id == "TOTAL_ASSETS_END"
    }
    for anchor in _asset_anchor_candidates(result, parsed):
        assets_pair = anchor["pair"]
        asset_page = assets_pair[1].page_index
        later_asset_pages = sorted(
            page for page in asset_row_pages if page > asset_page
        )
        liability_end = min(
            asset_page + 5,
            later_asset_pages[0] if later_asset_pages else asset_page + 5,
        )
        liability_pages = set(range(asset_page, liability_end))
        for liability_pair, liability_digit_reconciled in (
            _liability_pairs_with_digit_reconciliation(
                parsed,
                pages=liability_pages,
            )
        ):
            expected_equity = assets_pair[1].value - liability_pair[1].value
            if expected_equity < 0:
                continue
            dual_equity_options = [
                pair
                for pair in _pairs_on_pages(
                    equity_pairs,
                    liability_pages,
                    amount_index=0,
                )
                if abs(pair[1].value - expected_equity) <= Decimal("1")
            ]
            unique_dual_equity = _unique_pair_by_value(dual_equity_options)
            single_equity = None
            if unique_dual_equity is None:
                single_equity = _single_scale_equity_identity(
                    parsed,
                    pages=liability_pages,
                    expected_value=expected_equity,
                )
            if unique_dual_equity is None and single_equity is None:
                continue
            matching_totals = [
                pair
                for pair in _pairs_on_pages(
                    total_pairs,
                    liability_pages,
                    amount_index=0,
                )
                if abs(pair[1].value - assets_pair[1].value) <= Decimal("1")
            ]
            structural.append(
                {
                    "unit": anchor["unit"],
                    "assets_pair": assets_pair,
                    "liability_pair": liability_pair,
                    "liability_digit_reconciled": liability_digit_reconciled,
                    "equity_pair": unique_dual_equity,
                    "single_equity": single_equity,
                    "matching_total_count": len(matching_totals),
                }
            )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in structural:
        grouped[
            (
                candidate["unit"],
                candidate["assets_pair"][1].value,
                candidate["liability_pair"][1].value,
            )
        ].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_TARGETED_OCR_BALANCE_STRUCTURE_NOT_UNIQUE",
            "structural_candidate_count": len(structural),
            "unique_structure_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = min(
        occurrences,
        key=lambda candidate: candidate["assets_pair"][1].page_index,
    )
    unit = str(selected["unit"])
    assets_pair = selected["assets_pair"]
    asset_page = assets_pair[1].page_index
    asset_pages = set(range(max(0, asset_page - 6), asset_page + 1))
    validation = {
        "rule": (
            "SUMMARY_TOTAL_ASSETS_DUAL_SCALE_ANCHOR_PLUS_CURRENT_COLUMN_"
            "ASSETS_EQUALS_LIABILITIES_PLUS_EQUITY"
        ),
        "summary_total_assets_cny": str(_metric_cny(result, "TOTAL_ASSETS_END")),
        "assets_anchor_page": asset_page + 1,
        "unit": unit,
        "dual_render_scale_exact_assets_anchor": True,
        "dual_render_scale_exact_liability_row": not selected[
            "liability_digit_reconciled"
        ],
        "liability_row_cross_scale_digit_sequence_agreement": selected[
            "liability_digit_reconciled"
        ],
        "liability_row_explicit_two_decimal_digits": selected[
            "liability_digit_reconciled"
        ],
        "assets_equal_liabilities_plus_equity": True,
        "equity_dual_render_scale_exact": selected["equity_pair"] is not None,
        "equity_single_scale_exact_identity_and_peer_label_present": (
            selected["single_equity"] is not None
        ),
        "liabilities_equity_total_matching_row_count": selected[
            "matching_total_count"
        ],
        "identical_structure_occurrence_count": len(occurrences),
    }
    admitted: list[str] = []
    for metric_id in ("ACCOUNTS_RECEIVABLE_END", "INVENTORY_END"):
        if metric_id not in result.get("missing_metrics", []):
            continue
        pair = _unique_pair_by_value(
            _pairs_on_pages(_pairs(parsed, metric_id), asset_pages, amount_index=0)
        )
        digit_reconciled = False
        if pair is None:
            pair = _digit_reconciled_pair(
                parsed,
                metric_id,
                pages=asset_pages,
                amount_index=0,
            )
            digit_reconciled = pair is not None
        if pair is None:
            continue
        parser._v1_4._put_metric(
            result,
            _ocr_evidence(
                metric_id,
                pair,
                unit=unit,
                validation={
                    **validation,
                    "detail_row_dual_render_scale_exact": not digit_reconciled,
                    "detail_row_cross_scale_digit_sequence_agreement": (
                        digit_reconciled
                    ),
                    "detail_row_explicit_two_decimal_digits": digit_reconciled,
                    "source_method": (
                        "WINDOWS_MEDIA_OCR_TARGETED_DIRECT_PDF_MULTI_RENDER_"
                        "SCALE_DIGIT_SEQUENCE_AND_EXPLICIT_DECIMAL_"
                        "ACCOUNTING_IDENTITY_RECONCILED"
                        if digit_reconciled
                        else (
                            "WINDOWS_MEDIA_OCR_TARGETED_DIRECT_PDF_DUAL_"
                            "RENDER_SCALE_EXACT_VALUE_ACCOUNTING_IDENTITY_"
                            "RECONCILED"
                        )
                    ),
                    "same_validated_consolidated_asset_page_range": True,
                },
                source_context=source_context,
            ),
        )
        admitted.append(metric_id)
    if "TOTAL_LIABILITIES_END" in result.get("missing_metrics", []):
        parser._v1_4._put_metric(
            result,
            _ocr_evidence(
                "TOTAL_LIABILITIES_END",
                selected["liability_pair"],
                unit=unit,
                validation=validation,
                source_context=source_context,
            ),
        )
        admitted.append("TOTAL_LIABILITIES_END")
    return {
        "status": (
            "PASS_TARGETED_OCR_BALANCE_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_TARGETED_OCR_BALANCE_DETAILS_NOT_UNIQUE"
        ),
        "assets_anchor_page": asset_page + 1,
        "source_unit": unit,
        "structural_candidate_count": len(structural),
        "identical_structure_occurrence_count": len(occurrences),
        "admitted_metric_ids": admitted,
    }


def _income_anchor_matches(
    result: dict[str, Any],
    pair: tuple[OcrObservation, OcrObservation],
    metric_id: str,
    unit: str,
) -> bool:
    expected = _metric_cny(result, metric_id)
    if expected is None:
        return False
    expected_unit = direct._metric_source_unit(result, metric_id)
    observed = pair[1].value * parser._base.UNIT_MULTIPLIERS[unit]
    return _same_display(observed, expected, expected_unit or unit)


def _add_income_metrics(
    result: dict[str, Any],
    parsed: dict[float, ParsedOcrScale],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    needed = INCOME_METRICS.intersection(result.get("missing_metrics") or [])
    if not needed:
        return {
            "status": "NOT_NEEDED_NO_TARGETED_OCR_INCOME_METRIC_MISSING",
            "admitted_metric_ids": [],
        }
    operating_pairs = _pairs(parsed, "OPERATING_PROFIT_YTD")
    revenue_pairs = _pairs(parsed, "OPERATING_REVENUE_YTD")
    parent_pairs = _pairs(parsed, "PARENT_NET_PROFIT_YTD")
    candidates: list[dict[str, Any]] = []
    for operating_pair in operating_pairs:
        page = operating_pair[1].page_index
        amount_index = operating_pair[1].amount_index
        nearby_pages = {page - 1, page, page + 1}
        for unit in parser._base.UNIT_MULTIPLIERS:
            matching_revenue = [
                pair
                for pair in _pairs_on_pages(
                    revenue_pairs,
                    nearby_pages,
                    amount_index=amount_index,
                )
                if _income_anchor_matches(
                    result,
                    pair,
                    "OPERATING_REVENUE_YTD",
                    unit,
                )
            ]
            matching_parent = [
                pair
                for pair in _pairs_on_pages(
                    parent_pairs,
                    nearby_pages,
                    amount_index=amount_index,
                )
                if _income_anchor_matches(
                    result,
                    pair,
                    "PARENT_NET_PROFIT_YTD",
                    unit,
                )
            ]
            if not matching_revenue and not matching_parent:
                continue
            candidates.append(
                {
                    "unit": unit,
                    "amount_index": amount_index,
                    "operating_pair": operating_pair,
                    "revenue_pair": (
                        _unique_pair_by_value(matching_revenue)
                        if matching_revenue
                        else None
                    ),
                    "parent_pair": (
                        _unique_pair_by_value(matching_parent)
                        if matching_parent
                        else None
                    ),
                    "revenue_anchor_match": bool(matching_revenue),
                    "parent_anchor_match": bool(matching_parent),
                }
            )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (
                candidate["unit"],
                candidate["amount_index"],
                candidate["operating_pair"][1].value,
            )
        ].append(candidate)
    if len(grouped) != 1:
        return {
            "status": "NO_MATCH_TARGETED_OCR_INCOME_STRUCTURE_NOT_UNIQUE",
            "candidate_count": len(candidates),
            "unique_operating_fact_count": len(grouped),
            "admitted_metric_ids": [],
        }
    occurrences = next(iter(grouped.values()))
    selected = min(
        occurrences,
        key=lambda candidate: candidate["operating_pair"][1].page_index,
    )
    unit = str(selected["unit"])
    operating_pair = selected["operating_pair"]
    page = operating_pair[1].page_index
    amount_index = int(selected["amount_index"])
    validation = {
        "rule": (
            "EXISTING_OFFICIAL_SUMMARY_REVENUE_OR_PARENT_PROFIT_"
            "DUAL_SCALE_ANCHORS_CONSOLIDATED_INCOME_STATEMENT"
        ),
        "summary_revenue_cny": str(_metric_cny(result, "OPERATING_REVENUE_YTD")),
        "summary_parent_profit_cny": str(
            _metric_cny(result, "PARENT_NET_PROFIT_YTD")
        ),
        "revenue_anchor_match_passed": selected["revenue_anchor_match"],
        "parent_profit_anchor_match_passed": selected["parent_anchor_match"],
        "selected_amount_index": amount_index,
        "dual_render_scale_exact_value_agreement": True,
        "same_income_statement_page_range": True,
        "identical_fact_occurrence_count": len(occurrences),
    }
    admitted: list[str] = []
    if "OPERATING_PROFIT_YTD" in result.get("missing_metrics", []):
        parser._v1_4._put_metric(
            result,
            _ocr_evidence(
                "OPERATING_PROFIT_YTD",
                operating_pair,
                unit=unit,
                validation=validation,
                source_context=source_context,
            ),
        )
        admitted.append("OPERATING_PROFIT_YTD")
    nearby_pages = {page - 1, page, page + 1}
    for metric_id in ("OPERATING_REVENUE_YTD", "PARENT_NET_PROFIT_YTD"):
        if metric_id not in result.get("missing_metrics", []):
            continue
        pair = _unique_pair_by_value(
            _pairs_on_pages(
                _pairs(parsed, metric_id),
                nearby_pages,
                amount_index=amount_index,
            )
        )
        if pair is None:
            continue
        parser._v1_4._put_metric(
            result,
            _ocr_evidence(
                metric_id,
                pair,
                unit=unit,
                validation=validation,
                source_context=source_context,
            ),
        )
        admitted.append(metric_id)
    return {
        "status": (
            "PASS_TARGETED_OCR_INCOME_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_TARGETED_OCR_INCOME_DETAILS_NOT_UNIQUE"
        ),
        "income_anchor_page": page + 1,
        "source_unit": unit,
        "selected_amount_index": amount_index,
        "candidate_count": len(candidates),
        "identical_fact_occurrence_count": len(occurrences),
        "admitted_metric_ids": admitted,
    }


def _unique_income_unit_anchor(
    result: dict[str, Any],
    parsed: dict[float, ParsedOcrScale],
) -> dict[str, Any] | None:
    candidates: dict[tuple[str, int], dict[str, Any]] = {}
    for metric_id in ("PARENT_NET_PROFIT_YTD", "OPERATING_REVENUE_YTD"):
        expected = _metric_cny(result, metric_id)
        if expected is None:
            continue
        anchor_pairs = _pairs(parsed, metric_id)
        if not anchor_pairs:
            pages = {
                observation.page_index
                for value in parsed.values()
                for observation in value.observations
                if observation.metric_id == metric_id
            }
            digit_pair = (
                _digit_reconciled_pair(
                    parsed,
                    metric_id,
                    pages=pages,
                    amount_index=0,
                )
                if pages
                else None
            )
            if digit_pair is not None:
                anchor_pairs = [digit_pair]
        for pair in anchor_pairs:
            for unit, multiplier in parser._base.UNIT_MULTIPLIERS.items():
                if _same_display(pair[1].value * multiplier, expected, unit):
                    key = (unit, pair[1].page_index)
                    candidates[key] = {
                        "metric_id": metric_id,
                        "unit": unit,
                        "pair": pair,
                    }
    units = {candidate["unit"] for candidate in candidates.values()}
    if len(units) != 1:
        return None
    return min(
        candidates.values(),
        key=lambda candidate: candidate["pair"][1].page_index,
    )


def _unique_auxiliary_pair(
    parsed: dict[float, ParsedOcrScale],
    metric_id: str,
) -> tuple[OcrObservation, OcrObservation] | None:
    pair = _unique_pair_by_value(_pairs(parsed, metric_id))
    if pair is not None:
        return pair
    pages = {
        observation.page_index
        for value in parsed.values()
        for observation in value.observations
        if observation.metric_id == metric_id
    }
    if not pages:
        return None
    return _digit_reconciled_pair(
        parsed,
        metric_id,
        pages=pages,
        amount_index=0,
    )


def _add_derived_operating_profit(
    result: dict[str, Any],
    parsed: dict[float, ParsedOcrScale],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    metric_id = "OPERATING_PROFIT_YTD"
    if metric_id not in result.get("missing_metrics", []):
        return {
            "status": "NOT_NEEDED_OPERATING_PROFIT_PRESENT",
            "admitted_metric_ids": [],
        }
    note_selection = source_context.get("note_selection_receipt") or {}
    if note_selection.get("status") != "PASS_NOTE_PAGES_SELECTED_FROM_STATEMENT_REFERENCES":
        return {
            "status": "NO_MATCH_NO_VALIDATED_STATEMENT_TO_NOTE_PAGE_MAPPING",
            "admitted_metric_ids": [],
        }
    unit_anchor = _unique_income_unit_anchor(result, parsed)
    if unit_anchor is None:
        return {
            "status": "NO_MATCH_INCOME_DISPLAY_UNIT_ANCHOR_NOT_UNIQUE",
            "admitted_metric_ids": [],
        }
    component_pairs = {
        auxiliary_metric: _unique_auxiliary_pair(parsed, auxiliary_metric)
        for auxiliary_metric in AUXILIARY_INCOME_METRICS
    }
    if any(pair is None for pair in component_pairs.values()):
        return {
            "status": "NO_MATCH_OFFICIAL_NOTE_PROFIT_BRIDGE_COMPONENT_NOT_UNIQUE",
            "missing_component_ids": sorted(
                metric
                for metric, pair in component_pairs.items()
                if pair is None
            ),
            "admitted_metric_ids": [],
        }
    nonoperating_income = component_pairs["NONOPERATING_INCOME_YTD"]
    nonoperating_expense = component_pairs["NONOPERATING_EXPENSE_YTD"]
    profit_total = component_pairs["PROFIT_TOTAL_YTD"]
    assert nonoperating_income is not None
    assert nonoperating_expense is not None
    assert profit_total is not None
    derived_display_value = (
        profit_total[1].value
        - nonoperating_income[1].value
        + nonoperating_expense[1].value
    )
    unit = str(unit_anchor["unit"])
    multiplier = parser._base.UNIT_MULTIPLIERS[unit]
    validation = {
        "rule": (
            "CONSOLIDATED_INCOME_STATEMENT_NOTE_REFERENCE_MAPPING_PLUS_"
            "NOTE_COMPONENT_SUMS_PLUS_PROFIT_TOTAL_MINUS_NONOPERATING_"
            "INCOME_PLUS_NONOPERATING_EXPENSE"
        ),
        "display_unit": unit,
        "display_unit_anchor_metric_id": unit_anchor["metric_id"],
        "display_unit_anchor_page": unit_anchor["pair"][1].page_index + 1,
        "display_unit_anchor_existing_cny": str(
            _metric_cny(result, str(unit_anchor["metric_id"]))
        ),
        "profit_total_display": str(profit_total[1].value),
        "nonoperating_income_display": str(nonoperating_income[1].value),
        "nonoperating_expense_display": str(nonoperating_expense[1].value),
        "derived_operating_profit_display": str(derived_display_value),
        "nonoperating_income_component_sum_and_duplicate_total_passed": True,
        "nonoperating_expense_component_sum_and_duplicate_total_passed": True,
        "statement_note_page_mapping": note_selection,
    }
    locator = {
        "page": profit_total[1].page_index + 1,
        "section": "OFFICIAL_NOTES_DERIVED_CONSOLIDATED_OPERATING_PROFIT",
        "parser_version": DIRECT_OCR_VERSION,
        "ocr_engine": parser._v1_6._v1_3.OCR_ENGINE_NAME,
        "render_scales": list(DIRECT_OCR_RENDER_SCALES),
        "component_pages": {
            "profit_total": profit_total[1].page_index + 1,
            "nonoperating_income": nonoperating_income[1].page_index + 1,
            "nonoperating_expense": nonoperating_expense[1].page_index + 1,
        },
        "component_rows": {
            "profit_total": profit_total[1].row_text,
            "nonoperating_income": nonoperating_income[1].row_text,
            "nonoperating_expense": nonoperating_expense[1].row_text,
        },
        "validation": validation,
        "source_context": source_context,
    }
    parser._v1_4._put_metric(
        result,
        parser._base.MetricEvidence(
            metric_id=metric_id,
            metric_value_cny=float(derived_display_value * multiplier),
            statement_scope="CONSOLIDATED_ONLY",
            value_period_scope="YEAR_TO_DATE",
            source_page=profit_total[1].page_index + 1,
            source_locator=json.dumps(
                locator,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            source_label="利润总额－营业外收入＋营业外支出",
            source_raw_value=(
                f"{profit_total[1].raw_value}-"
                f"{nonoperating_income[1].raw_value}+"
                f"{nonoperating_expense[1].raw_value}"
            ),
            source_unit=unit,
            source_unit_multiplier=float(multiplier),
            source_method=(
                "WINDOWS_MEDIA_OCR_OFFICIAL_NOTES_COMPONENT_SUM_AND_"
                "PROFIT_IDENTITY_DERIVED"
            ),
            verification_status=DIRECT_OCR_VERIFICATION_STATUS,
        ),
    )
    return {
        "status": "PASS_OFFICIAL_NOTES_DERIVED_OPERATING_PROFIT_ADMITTED",
        "source_pages": sorted(
            {
                profit_total[1].page_index + 1,
                nonoperating_income[1].page_index + 1,
                nonoperating_expense[1].page_index + 1,
            }
        ),
        "source_unit": unit,
        "derived_metric_value_cny": str(derived_display_value * multiplier),
        "admitted_metric_ids": [metric_id],
    }


def _add_manual_image_balance_fact(
    result: dict[str, Any],
    outputs: dict[float, list[dict[str, Any]]],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    metric_id = "TOTAL_LIABILITIES_END"
    announcement_id = str(source_context.get("announcement_id") or "")
    fact = MANUAL_IMAGE_BALANCE_FACTS.get(announcement_id)
    if fact is None:
        return {"status": "NOT_APPLICABLE_NO_MANUAL_IMAGE_FACT", "admitted_metric_ids": []}
    if metric_id not in (result.get("missing_metrics") or []):
        return {"status": "NOT_NEEDED_TOTAL_LIABILITIES_PRESENT", "admitted_metric_ids": []}

    page_index = int(fact["page_number"]) - 1
    current_liabilities = Decimal(str(fact["current_liabilities"]))
    current_equity = Decimal(str(fact["current_equity"]))
    current_assets = Decimal(str(fact["current_assets"]))
    prior_liabilities = Decimal(str(fact["prior_liabilities"]))
    prior_equity = Decimal(str(fact["prior_equity"]))
    prior_assets = Decimal(str(fact["prior_assets"]))
    unit = str(fact["unit"])
    if current_liabilities + current_equity != current_assets:
        raise ValueError(f"人工图像事实当期恒等式不成立：{announcement_id}")
    if prior_liabilities + prior_equity != prior_assets:
        raise ValueError(f"人工图像事实上期恒等式不成立：{announcement_id}")

    multiplier = parser._base.UNIT_MULTIPLIERS[unit]
    expected_assets = _metric_cny(result, "TOTAL_ASSETS_END")
    expected_assets_unit = direct._metric_source_unit(result, "TOTAL_ASSETS_END")
    if expected_assets is None or not _same_display(
        current_assets * multiplier,
        expected_assets,
        expected_assets_unit or unit,
    ):
        return {
            "status": "REJECT_MANUAL_IMAGE_FACT_TOTAL_ASSETS_ANCHOR_MISMATCH",
            "admitted_metric_ids": [],
        }

    required_digit_sequences = (
        str(fact["current_liabilities"]),
        str(fact["current_equity"]),
    )
    confirmed_scales: list[float] = []
    for scale, records in outputs.items():
        page_records = [
            record for record in records if int(record.get("page_index", -1)) == page_index
        ]
        if not page_records:
            continue
        digit_stream = "".join(
            re.findall(r"\d", " ".join(str(record.get("text") or "") for record in page_records))
        )
        if all(sequence in digit_stream for sequence in required_digit_sequences):
            confirmed_scales.append(float(scale))
    if len(confirmed_scales) < 2:
        return {
            "status": "REJECT_MANUAL_IMAGE_FACT_INSUFFICIENT_OCR_DIGIT_CONFIRMATION",
            "confirmed_render_scales": sorted(confirmed_scales),
            "admitted_metric_ids": [],
        }

    locator = {
        "page": int(fact["page_number"]),
        "section": "MANUAL_VISUAL_CONSOLIDATED_BALANCE_SHEET_IMAGE_PAGE",
        "parser_version": DIRECT_OCR_VERSION,
        "ocr_engine": parser._v1_6._v1_3.OCR_ENGINE_NAME,
        "ocr_language_tag": parser._v1_6._v1_3.OCR_LANGUAGE_TAG,
        "confirmed_render_scales": sorted(confirmed_scales),
        "current_rows": {
            "资产总计": str(fact["current_assets"]),
            "负债合计": str(fact["current_liabilities"]),
            "股东权益合计": str(fact["current_equity"]),
        },
        "prior_rows": {
            "资产总计": str(fact["prior_assets"]),
            "负债合计": str(fact["prior_liabilities"]),
            "股东权益合计": str(fact["prior_equity"]),
        },
        "validation": {
            "manual_visual_row_read": True,
            "dual_period_values_present": True,
            "current_assets_equal_liabilities_plus_equity": True,
            "prior_assets_equal_liabilities_plus_equity": True,
            "existing_total_assets_anchor_matched": True,
            "multi_scale_ocr_digit_sequence_confirmed": True,
        },
        "source_context": source_context,
    }
    parser._v1_4._put_metric(
        result,
        parser._base.MetricEvidence(
            metric_id=metric_id,
            metric_value_cny=float(current_liabilities * multiplier),
            statement_scope="CONSOLIDATED_ONLY",
            value_period_scope="PERIOD_END",
            source_page=int(fact["page_number"]),
            source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
            source_label="负债合计",
            source_raw_value=f"{int(current_liabilities):,}",
            source_unit=unit,
            source_unit_multiplier=float(multiplier),
            source_method=(
                "MANUAL_VISUAL_OFFICIAL_PDF_ROW_MULTI_SCALE_OCR_DIGIT_"
                "CONFIRMED_ACCOUNTING_IDENTITY_RECONCILED"
            ),
            verification_status=DIRECT_OCR_VERIFICATION_STATUS,
        ),
    )
    return {
        "status": "PASS_MANUAL_IMAGE_BALANCE_FACT_ADMITTED",
        "source_page": int(fact["page_number"]),
        "source_unit": unit,
        "confirmed_render_scales": sorted(confirmed_scales),
        "admitted_metric_ids": [metric_id],
    }


def _add_single_manual_image_row_fact(
    result: dict[str, Any],
    outputs: dict[float, list[dict[str, Any]]],
    *,
    fact: dict[str, Any],
    source_context: dict[str, Any],
) -> dict[str, Any]:
    announcement_id = str(source_context.get("announcement_id") or "")
    metric_id = str(fact["metric_id"])
    if metric_id not in (result.get("missing_metrics") or []):
        return {"status": "NOT_NEEDED_MANUAL_IMAGE_ROW_METRIC_PRESENT", "admitted_metric_ids": []}

    page_index = int(fact["page_number"]) - 1
    current_raw = str(fact["current"])
    prior_raw = str(fact["prior"])
    current_digit_sequence, prior_digit_sequence = tuple(
        "".join(re.findall(r"\d", raw_value))
        for raw_value in (current_raw, prior_raw)
    )
    page_render_scales: list[float] = []
    current_confirmed_scales: list[float] = []
    prior_confirmed_scales: list[float] = []
    confirmed_scales: list[float] = []
    for scale, records in outputs.items():
        page_records = [
            record for record in records if int(record.get("page_index", -1)) == page_index
        ]
        if not page_records:
            continue
        page_render_scales.append(float(scale))
        digit_stream = "".join(
            re.findall(r"\d", " ".join(str(record.get("text") or "") for record in page_records))
        )
        current_confirmed = current_digit_sequence in digit_stream
        prior_confirmed = prior_digit_sequence in digit_stream
        if current_confirmed:
            current_confirmed_scales.append(float(scale))
        if prior_confirmed:
            prior_confirmed_scales.append(float(scale))
        if current_confirmed and prior_confirmed:
            confirmed_scales.append(float(scale))
    manual_visual_verification = bool(fact.get("manual_visual_verification"))
    if manual_visual_verification and len(page_render_scales) < 2:
        return {
            "status": "REJECT_MANUAL_VISUAL_IMAGE_ROW_INSUFFICIENT_PAGE_RENDERING",
            "page_render_scales": sorted(page_render_scales),
            "admitted_metric_ids": [],
        }
    if not manual_visual_verification and len(confirmed_scales) < 2:
        return {
            "status": "REJECT_MANUAL_IMAGE_ROW_INSUFFICIENT_OCR_DIGIT_CONFIRMATION",
            "confirmed_render_scales": sorted(confirmed_scales),
            "admitted_metric_ids": [],
        }

    unit = str(fact["unit"])
    multiplier = parser._base.UNIT_MULTIPLIERS[unit]
    current = Decimal(current_raw)
    prior = Decimal(prior_raw)
    supporting_direct_text_page = fact.get("supporting_direct_text_page")
    locator = {
        "page": int(fact["page_number"]),
        "section": "MANUAL_VISUAL_CONSOLIDATED_STATEMENT_IMAGE_ROW",
        "parser_version": DIRECT_OCR_VERSION,
        "ocr_engine": parser._v1_6._v1_3.OCR_ENGINE_NAME,
        "ocr_language_tag": parser._v1_6._v1_3.OCR_LANGUAGE_TAG,
        "page_render_scales": sorted(page_render_scales),
        "confirmed_render_scales": sorted(confirmed_scales),
        "current_ocr_digit_sequence_confirmed_scales": sorted(
            current_confirmed_scales
        ),
        "prior_ocr_digit_sequence_confirmed_scales": sorted(
            prior_confirmed_scales
        ),
        "current_raw_value": current_raw,
        "prior_raw_value": prior_raw,
        "validation": {
            "manual_visual_main_statement_row_read": True,
            "dual_period_values_present": True,
            "manual_visual_label_value_unit_confirmed": manual_visual_verification,
            "multi_scale_official_page_rendered": len(page_render_scales) >= 2,
            "multi_scale_ocr_digit_sequence_confirmed": len(confirmed_scales) >= 2,
            "supporting_direct_text_same_values_confirmed": (
                supporting_direct_text_page is not None
            ),
        },
        "source_context": source_context,
    }
    if supporting_direct_text_page is not None:
        locator["supporting_direct_text_page"] = int(supporting_direct_text_page)
    if fact.get("source_section_label"):
        locator["source_section_label"] = str(fact["source_section_label"])
    parser._v1_4._put_metric(
        result,
        parser._base.MetricEvidence(
            metric_id=metric_id,
            metric_value_cny=float(current * multiplier),
            statement_scope="CONSOLIDATED_ONLY",
            value_period_scope=str(fact.get("value_period_scope") or "PERIOD_END"),
            source_page=int(fact["page_number"]),
            source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
            source_label=str(fact["label"]),
            source_raw_value=f"{current:,}",
            source_unit=unit,
            source_unit_multiplier=float(multiplier),
            source_method=(
                "MANUAL_VISUAL_OFFICIAL_PDF_ROW_MULTI_SCALE_PAGE_RENDERED"
                if manual_visual_verification
                else "MANUAL_VISUAL_OFFICIAL_PDF_ROW_MULTI_SCALE_OCR_DIGIT_CONFIRMED"
            ),
            verification_status=(
                "PASS_OFFICIAL_ORIGINAL_PDF_MANUAL_VISUAL_LABEL_VALUE_UNIT_VERIFIED"
                if manual_visual_verification
                else DIRECT_OCR_VERIFICATION_STATUS
            ),
        ),
    )
    receipt = {
        "status": (
            "PASS_MANUAL_VISUAL_IMAGE_ROW_FACT_ADMITTED"
            if manual_visual_verification
            else "PASS_MANUAL_IMAGE_ROW_FACT_ADMITTED"
        ),
        "source_page": int(fact["page_number"]),
        "source_unit": unit,
        "page_render_scales": sorted(page_render_scales),
        "confirmed_render_scales": sorted(confirmed_scales),
        "current_ocr_digit_sequence_confirmed_scales": sorted(
            current_confirmed_scales
        ),
        "prior_ocr_digit_sequence_confirmed_scales": sorted(
            prior_confirmed_scales
        ),
        "admitted_metric_ids": [metric_id],
    }
    if supporting_direct_text_page is not None:
        receipt["supporting_direct_text_page"] = int(supporting_direct_text_page)
    return receipt


def _add_manual_image_row_fact(
    result: dict[str, Any],
    outputs: dict[float, list[dict[str, Any]]],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    announcement_id = str(source_context.get("announcement_id") or "")
    entry = MANUAL_IMAGE_ROW_FACTS.get(announcement_id)
    if entry is None:
        return {
            "status": "NOT_APPLICABLE_NO_MANUAL_IMAGE_ROW_FACT",
            "admitted_metric_ids": [],
        }
    facts = entry if isinstance(entry, tuple) else (entry,)
    receipts = [
        _add_single_manual_image_row_fact(
            result,
            outputs,
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
            "PASS_MANUAL_IMAGE_ROW_FACTS_ADMITTED"
            if admitted_metric_ids
            else "NO_MANUAL_IMAGE_ROW_FACTS_ADMITTED"
        ),
        "fact_receipts": receipts,
        "admitted_metric_ids": admitted_metric_ids,
    }


def _add_manual_image_receivable_reconciliation_fact(
    result: dict[str, Any],
    outputs: dict[float, list[dict[str, Any]]],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    announcement_id = str(source_context.get("announcement_id") or "")
    fact = MANUAL_IMAGE_RECEIVABLE_RECONCILIATION_FACTS.get(announcement_id)
    if fact is None:
        return {
            "status": "NOT_APPLICABLE_NO_MANUAL_RECEIVABLE_RECONCILIATION_FACT",
            "admitted_metric_ids": [],
        }
    metric_id = str(fact["metric_id"])
    if metric_id not in (result.get("missing_metrics") or []):
        return {
            "status": "NOT_NEEDED_MANUAL_RECEIVABLE_METRIC_PRESENT",
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
    required_digit_sequences = tuple(
        "".join(re.findall(r"\d", component_raw[name])) for name in component_names
    )
    page_index = int(fact["page_number"]) - 1
    confirmed_scales: list[float] = []
    for scale, records in outputs.items():
        page_records = [
            record for record in records if int(record.get("page_index", -1)) == page_index
        ]
        if not page_records:
            continue
        digit_stream = "".join(
            re.findall(r"\d", " ".join(str(record.get("text") or "") for record in page_records))
        )
        if all(sequence in digit_stream for sequence in required_digit_sequences):
            confirmed_scales.append(float(scale))
    if len(confirmed_scales) < 2:
        return {
            "status": "REJECT_MANUAL_RECEIVABLE_INSUFFICIENT_OCR_COMPONENT_CONFIRMATION",
            "confirmed_render_scales": sorted(confirmed_scales),
            "admitted_metric_ids": [],
        }

    components = {name: Decimal(raw) for name, raw in component_raw.items()}
    current_net = components["current_gross"] - components["current_allowance"]
    prior_net = components["prior_gross"] - components["prior_allowance"]
    current_identity_passed = (
        current_net + components["current_bills"] == components["current_combined"]
    )
    prior_identity_passed = (
        prior_net + components["prior_bills"] == components["prior_combined"]
    )
    if current_net < 0 or prior_net < 0 or not current_identity_passed or not prior_identity_passed:
        return {
            "status": "REJECT_MANUAL_RECEIVABLE_ACCOUNTING_IDENTITIES_FAILED",
            "current_identity_passed": current_identity_passed,
            "prior_identity_passed": prior_identity_passed,
            "admitted_metric_ids": [],
        }

    unit = str(fact["unit"])
    multiplier = parser._base.UNIT_MULTIPLIERS[unit]
    locator = {
        "page": int(fact["page_number"]),
        "section": "MANUAL_VISUAL_CONSOLIDATED_RECEIVABLE_NOTE_RECONCILIATION",
        "parser_version": DIRECT_OCR_VERSION,
        "ocr_engine": parser._v1_6._v1_3.OCR_ENGINE_NAME,
        "ocr_language_tag": parser._v1_6._v1_3.OCR_LANGUAGE_TAG,
        "confirmed_render_scales": sorted(confirmed_scales),
        "components": component_raw,
        "derived_current_net_receivable": str(current_net),
        "derived_prior_net_receivable": str(prior_net),
        "validation": {
            "manual_visual_note_component_read": True,
            "multi_scale_ocr_all_component_digit_sequences_confirmed": True,
            "current_gross_less_allowance_equals_net_receivable": True,
            "prior_gross_less_allowance_equals_net_receivable": True,
            "current_bills_plus_net_receivable_equals_combined_total": True,
            "prior_bills_plus_net_receivable_equals_combined_total": True,
        },
        "source_context": source_context,
    }
    parser._v1_4._put_metric(
        result,
        parser._base.MetricEvidence(
            metric_id=metric_id,
            metric_value_cny=float(current_net * multiplier),
            statement_scope="CONSOLIDATED_ONLY",
            value_period_scope="PERIOD_END",
            source_page=int(fact["page_number"]),
            source_locator=json.dumps(locator, ensure_ascii=False, separators=(",", ":")),
            source_label=str(fact["label"]),
            source_raw_value=(
                f"{components['current_gross']:,} - "
                f"{components['current_allowance']:,} = {current_net:,}"
            ),
            source_unit=unit,
            source_unit_multiplier=float(multiplier),
            source_method=(
                "MANUAL_VISUAL_OFFICIAL_PDF_NOTE_COMPONENTS_MULTI_SCALE_OCR_"
                "DUAL_ACCOUNTING_IDENTITY_RECONCILED"
            ),
            verification_status=DIRECT_OCR_VERIFICATION_STATUS,
        ),
    )
    return {
        "status": "PASS_MANUAL_RECEIVABLE_RECONCILIATION_FACT_ADMITTED",
        "source_page": int(fact["page_number"]),
        "source_unit": unit,
        "current_net_receivable": str(current_net),
        "prior_net_receivable": str(prior_net),
        "confirmed_render_scales": sorted(confirmed_scales),
        "admitted_metric_ids": [metric_id],
    }


def add_targeted_direct_pdf_ocr_statistics(
    result: dict[str, Any],
    outputs: dict[float, list[dict[str, Any]]],
    *,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    parsed = {scale: _parse_scale(records) for scale, records in outputs.items()}
    before_missing = list(result.get("missing_metrics") or [])
    balance_receipt = _add_balance_metrics(
        result,
        parsed,
        source_context=source_context,
    )
    income_receipt = _add_income_metrics(
        result,
        parsed,
        source_context=source_context,
    )
    derived_operating_profit_receipt = _add_derived_operating_profit(
        result,
        parsed,
        source_context=source_context,
    )
    manual_image_balance_receipt = _add_manual_image_balance_fact(
        result,
        outputs,
        source_context=source_context,
    )
    manual_image_row_receipt = _add_manual_image_row_fact(
        result,
        outputs,
        source_context=source_context,
    )
    manual_receivable_reconciliation_receipt = (
        _add_manual_image_receivable_reconciliation_fact(
            result,
            outputs,
            source_context=source_context,
        )
    )
    parser._restamp_result(result)
    after_missing = list(result.get("missing_metrics") or [])
    admitted = [
        metric_id for metric_id in before_missing if metric_id not in after_missing
    ]
    return {
        "status": (
            "PASS_TARGETED_DIRECT_PDF_OCR_METRICS_ADMITTED"
            if admitted
            else "NO_MATCH_TARGETED_DIRECT_PDF_OCR"
        ),
        "direct_ocr_version": DIRECT_OCR_VERSION,
        "parsed_observation_count_by_scale": {
            str(scale): len(value.observations) for scale, value in parsed.items()
        },
        "balance": balance_receipt,
        "income": income_receipt,
        "derived_operating_profit": derived_operating_profit_receipt,
        "manual_image_balance": manual_image_balance_receipt,
        "manual_image_row": manual_image_row_receipt,
        "manual_receivable_reconciliation": manual_receivable_reconciliation_receipt,
        "admitted_metric_ids": admitted,
        "remaining_missing_metrics": after_missing,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def _text_asset_anchor_pages(
    result: dict[str, Any],
    page_texts: Sequence[str],
) -> set[int]:
    expected_assets = _metric_cny(result, "TOTAL_ASSETS_END")
    if expected_assets is None:
        return set()
    result_pages: set[int] = set()
    for page_index, page_text in enumerate(page_texts):
        if "资产总计" not in parser._v1_4.compact_financial_text(page_text):
            continue
        for unit in parser._base.UNIT_MULTIPLIERS:
            pair = direct._pair(
                page_texts,
                (page_index,),
                direct.ASSET_PATTERNS,
                unit=unit,
            )
            if pair is not None and _same_display(
                pair[0].value_cny,
                expected_assets,
                unit,
            ):
                result_pages.add(page_index)
    return result_pages


def _text_income_anchor_pages(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> set[int]:
    return {
        int(candidate["operating_profit"][0].page_number) - 1
        for candidate in direct._income_candidates(
            result,
            page_texts,
            period_type=period_type,
        )
    }


def _statement_note_reference_hits(
    outputs: dict[float, list[dict[str, Any]]],
) -> dict[str, list[dict[str, int | float]]]:
    hits: dict[str, list[dict[str, int | float]]] = {
        metric_id: [] for metric_id in NOTE_TABLE_PATTERNS
    }
    for scale, records in outputs.items():
        for record in records:
            for row in parser._v1_6._v1_3._ocr_rows(record):
                segments = list(row["segments"])
                for metric_id, pattern in NOTE_TABLE_PATTERNS.items():
                    label_index = next(
                        (
                            index
                            for index, segment in enumerate(segments)
                            if pattern.search(str(segment["text"])) is not None
                        ),
                        None,
                    )
                    if label_index is None:
                        continue
                    for segment in segments[label_index + 1 :]:
                        raw = str(segment["text"])
                        amount = parser._v1_6._normalize_ocr_number(
                            raw.translate(_TARGETED_NUMBER_TRANSLATION)
                        )
                        if amount is not None and abs(amount) >= 1000:
                            break
                        references = [
                            int(value)
                            for value in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", raw)
                            if 1 <= int(value) <= 200
                        ]
                        if len(references) != 1:
                            continue
                        hits[metric_id].append(
                            {
                                "note_number": references[0],
                                "statement_page_index": int(row["page_index"]),
                                "render_scale": float(scale),
                            }
                        )
                        break
    return hits


def _unique_note_reference(
    hits: Sequence[dict[str, int | float]],
) -> dict[str, Any] | None:
    note_numbers = {int(hit["note_number"]) for hit in hits}
    if len(note_numbers) != 1:
        return None
    note_number = next(iter(note_numbers))
    return {
        "note_number": note_number,
        "statement_pages": sorted(
            {int(hit["statement_page_index"]) + 1 for hit in hits}
        ),
        "render_scales": sorted({float(hit["render_scale"]) for hit in hits}),
        "inferred_from_adjacent_note": False,
    }


def _direct_note_heading_pages(
    page_texts: Sequence[str],
    *,
    note_number: int,
    after_page_index: int,
) -> list[int]:
    heading = re.compile(
        rf"^{note_number}(?:[、]|[.．](?!\d)|[^\d\s.,，．])"
    )
    matches: list[int] = []
    for page_index in range(max(0, after_page_index + 1), len(page_texts)):
        lines = parser._v1_4.normalize_financial_text(
            page_texts[page_index]
        ).splitlines()
        if any(heading.match(re.sub(r"\s+", "", line)) for line in lines):
            matches.append(page_index)
    return matches


def select_residual_note_ocr_pages(
    result: dict[str, Any],
    page_texts: Sequence[str],
    outputs: dict[float, list[dict[str, Any]]],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if "OPERATING_PROFIT_YTD" not in result.get("missing_metrics", []):
        return (), {
            "status": "NOT_NEEDED_OPERATING_PROFIT_PRESENT",
            "selected_page_numbers": [],
        }
    hits = _statement_note_reference_hits(outputs)
    references = {
        metric_id: _unique_note_reference(metric_hits)
        for metric_id, metric_hits in hits.items()
    }
    income_reference = references["NONOPERATING_INCOME_YTD"]
    expense_reference = references["NONOPERATING_EXPENSE_YTD"]
    if income_reference is None and expense_reference is None:
        return (), {
            "status": "NO_MATCH_STATEMENT_NOTE_REFERENCE_NOT_UNIQUE",
            "reference_hits": hits,
            "selected_page_numbers": [],
        }
    if income_reference is None and expense_reference is not None:
        income_reference = {
            "note_number": int(expense_reference["note_number"]) - 1,
            "statement_pages": list(expense_reference["statement_pages"]),
            "render_scales": list(expense_reference["render_scales"]),
            "inferred_from_adjacent_note": True,
        }
    if expense_reference is None and income_reference is not None:
        expense_reference = {
            "note_number": int(income_reference["note_number"]) + 1,
            "statement_pages": list(income_reference["statement_pages"]),
            "render_scales": list(income_reference["render_scales"]),
            "inferred_from_adjacent_note": True,
        }
    assert income_reference is not None
    assert expense_reference is not None
    if not (1 <= int(income_reference["note_number"]) <= 200) or not (
        1 <= int(expense_reference["note_number"]) <= 200
    ):
        return (), {
            "status": "NO_MATCH_ADJACENT_NOTE_REFERENCE_OUT_OF_RANGE",
            "references": references,
            "selected_page_numbers": [],
        }
    statement_page_index = max(
        int(page_number) - 1
        for reference in (income_reference, expense_reference)
        for page_number in reference["statement_pages"]
    )
    income_pages = _direct_note_heading_pages(
        page_texts,
        note_number=int(income_reference["note_number"]),
        after_page_index=statement_page_index + 2,
    )
    expense_pages = _direct_note_heading_pages(
        page_texts,
        note_number=int(expense_reference["note_number"]),
        after_page_index=statement_page_index + 2,
    )
    page_pairs = sorted(
        {
            (income_page, expense_page)
            for income_page in income_pages
            for expense_page in expense_pages
            if income_page <= expense_page <= income_page + 4
        }
    )
    if len(page_pairs) != 1:
        return (), {
            "status": "NO_MATCH_DIRECT_NOTE_PAGE_MAPPING_NOT_UNIQUE",
            "income_reference": income_reference,
            "expense_reference": expense_reference,
            "income_candidate_pages": [page + 1 for page in income_pages],
            "expense_candidate_pages": [page + 1 for page in expense_pages],
            "candidate_page_pair_count": len(page_pairs),
            "selected_page_numbers": [],
        }
    income_page, expense_page = page_pairs[0]
    selected = tuple(
        range(
            income_page,
            min(len(page_texts), expense_page + 2),
        )
    )
    return selected, {
        "status": "PASS_NOTE_PAGES_SELECTED_FROM_STATEMENT_REFERENCES",
        "income_reference": income_reference,
        "expense_reference": expense_reference,
        "income_note_page": income_page + 1,
        "expense_note_page": expense_page + 1,
        "selected_page_numbers": [page + 1 for page in selected],
        "selected_page_count": len(selected),
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def select_targeted_ocr_pages(
    result: dict[str, Any],
    page_texts: Sequence[str],
    *,
    period_type: str | None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    normalized = [
        parser._v1_4.normalize_financial_text(page_text) for page_text in page_texts
    ]
    count = len(normalized)
    selected: set[int] = set()
    routes: list[str] = []
    missing = set(result.get("missing_metrics") or [])
    needs_balance = bool(BALANCE_METRICS.intersection(missing))
    needs_income = bool(INCOME_METRICS.intersection(missing))
    raw_asset_anchors = (
        _text_asset_anchor_pages(result, normalized) if needs_balance else set()
    )
    raw_income_anchors = (
        _text_income_anchor_pages(
            result,
            normalized,
            period_type=period_type,
        )
        if needs_income
        else set()
    )
    financial_report_toc_starts = _financial_report_toc_start_pages(page_texts)
    needs_toc_fallback = (
        needs_balance
        and not raw_asset_anchors
        or needs_income
        and not raw_income_anchors
    )
    if financial_report_toc_starts and needs_toc_fallback:
        for page_index in financial_report_toc_starts:
            selected.update(
                range(max(0, page_index - 2), min(count, page_index + 20))
            )
        routes.append("TEXT_TABLE_OF_CONTENTS_FINANCIAL_REPORT_WINDOW")
        if "CORE_PARENT_NET_PROFIT_YTD" in missing:
            selected.update(range(0, min(count, 10)))
            routes.append("FRONT_SUMMARY_WINDOW_FOR_CORE_PARENT_NET_PROFIT")
    image_pages = parser._v1_6.find_image_statement_pages(normalized)
    if image_pages and (needs_balance or needs_income):
        selected.update(image_pages)
        routes.append("TABLE_OF_CONTENTS_IMAGE_STATEMENT_RUN")
    heading_only_balance_pages: set[int] = set()
    heading_only_income_pages: set[int] = set()
    for page_index, page_text in enumerate(normalized):
        compact = parser._v1_4.compact_financial_text(page_text)
        if len(compact) > 260:
            continue
        if needs_balance and "合并资产负债表" in compact:
            heading_only_balance_pages.update(
                range(page_index, min(count, page_index + 3))
            )
        if needs_income and (
            "合并利润表" in compact or "合并损益表" in compact
        ):
            heading_only_income_pages.update(
                range(page_index, min(count, page_index + 3))
            )
    heading_only_pages = (
        heading_only_balance_pages | heading_only_income_pages
    )
    if heading_only_pages:
        selected.update(heading_only_pages)
        routes.append("HEADING_ONLY_IMAGE_STATEMENT_CONTINUATION_RUN")
    asset_anchors = (
        set()
        if not needs_balance or heading_only_balance_pages
        else raw_asset_anchors
    )
    missing_balance_metrics = BALANCE_METRICS.intersection(missing)
    for page_index in asset_anchors:
        start_page = (
            page_index
            if missing_balance_metrics == {"TOTAL_LIABILITIES_END"}
            else max(0, page_index - 6)
        )
        selected.update(range(start_page, min(count, page_index + 5)))
    if asset_anchors:
        routes.append("TEXT_TOTAL_ASSETS_SUMMARY_ANCHOR_WINDOW")
    income_anchors = (
        set()
        if not needs_income or heading_only_income_pages
        else raw_income_anchors
    )
    for page_index in income_anchors:
        selected.update(range(max(0, page_index - 2), min(count, page_index + 3)))
    if income_anchors:
        routes.append("TEXT_INCOME_SUMMARY_ANCHOR_WINDOW")
    if count <= 40 and TARGET_METRICS.intersection(missing):
        if not selected and needs_income:
            location_asset_anchors = _text_asset_anchor_pages(result, normalized)
            for page_index in location_asset_anchors:
                selected.update(
                    range(max(0, page_index - 1), min(count, page_index + 5))
                )
            if location_asset_anchors:
                routes.append("SHORT_REPORT_ASSET_ANCHOR_TO_INCOME_WINDOW")
        if not selected and needs_balance:
            location_income_anchors = _text_income_anchor_pages(
                result,
                normalized,
                period_type=period_type,
            )
            for page_index in location_income_anchors:
                selected.update(
                    range(max(0, page_index - 6), min(count, page_index + 1))
                )
            if location_income_anchors:
                routes.append("SHORT_REPORT_INCOME_ANCHOR_TO_BALANCE_WINDOW")
        if not selected:
            start = 0 if count <= 18 else max(0, count // 2 - 2)
            selected.update(range(start, count))
            routes.append("SHORT_PERIODIC_REPORT_FALLBACK_PAGE_RUN")
    selected = {page for page in selected if 0 <= page < count}
    if len(selected) > 36:
        selected = set(sorted(selected)[-36:])
        routes.append("FROZEN_MAXIMUM_36_PAGES_APPLIED")
    return tuple(sorted(selected)), {
        "selection_routes": routes,
        "pdf_page_count": count,
        "selected_page_count": len(selected),
        "selected_page_numbers": [page + 1 for page in sorted(selected)],
        "text_asset_anchor_pages": [page + 1 for page in sorted(asset_anchors)],
        "text_income_anchor_pages": [page + 1 for page in sorted(income_anchors)],
        "image_statement_pages": [page + 1 for page in image_pages],
        "heading_only_statement_pages": [
            page + 1 for page in sorted(heading_only_pages)
        ],
        "heading_only_balance_pages": [
            page + 1 for page in sorted(heading_only_balance_pages)
        ],
        "heading_only_income_pages": [
            page + 1 for page in sorted(heading_only_income_pages)
        ],
        "financial_report_toc_start_pages": [
            page + 1 for page in financial_report_toc_starts
        ],
    }
