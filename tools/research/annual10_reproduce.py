"""验证交付输入，或在新目录重做六阶段研究；不会连接券商或改写原工作区。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = Path('research_runs/annual10_sharpe12_20260910')
INPUTS = RUN / 'reproduction_inputs.json'
PHASES = ('capital', 'residual', 'episode_value', 'daily_opportunity', 'etf_flow', 'winning_budget')


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(root: Path) -> list[dict]:
    rows = json.loads((root / INPUTS).read_text(encoding='utf-8'))['files']
    for row in rows:
        path = root / row['path']
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError(f"输入路径非法：{row['path']}")
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError(f"输入摘要不符：{row['path']}")
    print(json.dumps({'verified_inputs': len(rows), 'status': 'PASS'}, ensure_ascii=False), flush=True)
    return rows


def prepare(destination: Path) -> Path:
    rows = validate(ROOT)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('复现目录必须尚不存在；不覆盖任何旧文件。')
    destination.mkdir(parents=True)
    for relative in [row['path'] for row in rows] + [INPUTS.as_posix(), 'tools/research/annual10_reproduce.py', (RUN / 'requirements-reproduce.txt').as_posix()]:
        source, target = ROOT / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    validate(destination)
    print(f'新复现工作区：{destination}', flush=True)
    return destination


def invoke(root: Path, script: str, *args: str) -> None:
    env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(root / 'tools/research' / script), *args]
    print('执行：' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def scenarios(root: Path, script: str) -> None:
    for period in ('evaluation', 'earlier_diagnostic'):
        for cost in ('BASE', 'STRESS'):
            invoke(root, script, period, cost)


def rerun(destination: Path) -> None:
    root = prepare(destination)
    invoke(root, 'annual10_capital_20260910.py')
    scenarios(root, 'annual10_residual_20260910.py')
    for phase in ('episode_value', 'daily_opportunity'):
        script = f'annual10_{phase}_20260910.py'
        invoke(root, script, 'prepare')
        invoke(root, script, 'fit')
        if phase == 'daily_opportunity':
            # 原运行经中断恢复后用completed_model_fits字段；完整重跑只补同义统计字段。
            path = root / RUN / phase / 'training_records.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['completed_model_fits'] = data['actual_fits']
            data['reproduction_note'] = 'Uninterrupted full fit; count alias only. Predictions and policies unchanged.'
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        scenarios(root, script)
    invoke(root, 'annual10_etf_flow_20260910.py', 'prepare')
    for horizon in ('5', '20'):
        invoke(root, 'annual10_etf_flow_20260910.py', 'fit', horizon)
    scenarios(root, 'annual10_etf_flow_20260910.py')
    scenarios(root, 'annual10_winning_budget_20260910.py')
    invoke(root, 'annual10_validate_20260910.py')
    for phase in PHASES:
        invoke(root, 'annual10_validate_20260910.py', 'accounts', phase)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *[
        f'tests/{name}' for name in (
            'test_event_clock_account_v1.py', 'test_two_policy_min_variance_v1.py',
            'test_monotone_episode_budget_v1.py', 'test_trend_noise_reference_blend_v1.py',
            'test_rearmed_cycle_exit_account_v1.py')]], cwd=root, check=True)
    invoke(root, 'annual10_summarize_20260910.py')
    # 只比较保存的实际绩效；时间戳和跨平台Parquet字节不被当作策略差异。
    import numpy as np
    import pandas as pd
    key = ['phase', 'model', 'period', 'cost']
    left = pd.read_csv(ROOT / RUN / 'all_metrics.csv').set_index(key).sort_index()
    right = pd.read_csv(root / RUN / 'all_metrics.csv').set_index(key).sort_index()
    if not left.index.equals(right.index):
        raise ValueError('复现候选或情景集合不一致。')
    columns = ['annualized_return', 'net_sharpe', 'max_drawdown', 'mean_exposure', 'trade_count', 'commission', 'slippage_cost']
    np.testing.assert_allclose(left[columns].to_numpy(float), right[columns].to_numpy(float), rtol=1e-8, atol=1e-10, equal_nan=True)
    print(json.dumps({'reproduced_policy_accounts': len(right), 'metrics_equal': True,
                      'independent_market_evidence_added': False}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'prepare', 'rerun'])
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        validate(ROOT)
    elif args.destination is None:
        parser.error('prepare/rerun必须指定尚不存在的--destination。')
    elif args.action == 'prepare':
        prepare(args.destination)
    else:
        rerun(args.destination)


if __name__ == '__main__':
    main()
