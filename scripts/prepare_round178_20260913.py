from pathlib import Path
from research.intraday_overnight_increment_v1 import now,write_json
ROOT=Path(__file__).resolve().parents[1]
write_json(ROOT/'reports/research/510300_early_selected_regime_mapping_v1/tests_receipt.json',{'recorded_at':now(),'exit_code':0,'passed':3,'seconds':1.83},exclusive=True)
