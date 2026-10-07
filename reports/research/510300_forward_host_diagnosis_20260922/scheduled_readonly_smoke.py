"""仅由临时计划任务调用原入口的校验模式；不采集行情。"""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[3]
interpreter = root / '.venv/Scripts/python.exe'
command = [str(interpreter), '-X', 'utf8', '-m', 'scripts.run_priority_forward_codex_automation_v1_11_1', '--verify-only', '--expected-manifest-sha256', '2ffef04f1edcba3171a7f5f83e84c99ea0bdbbafaedbc194e6683ce954b40a52']
try:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
    payload = {'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'collection_triggered': False, 'quality_day_added': False}
except Exception as exc:
    payload = {'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'status': 'SMOKE_FAILED', 'error': str(exc), 'collection_triggered': False, 'quality_day_added': False}
Path(__file__).with_name('scheduled_readonly_smoke_result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')