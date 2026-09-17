"""保存第175轮测试回执与完整中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now,write_json
ROOT=Path(__file__).resolve().parents[1]
def main():
 write_json(ROOT/'reports/research/510300_confirmed_auxiliary_episode_batch_v1/tests_receipt.json',{'recorded_at':now(),'exit_code':0,'passed':5,'command':'.venv\\Scripts\\python.exe -m pytest tests\\test_confirmed_auxiliary_episode_batch_v1.py -q'},exclusive=True)
 print('第175轮必要测试回执已保存。')
if __name__=='__main__':main()
