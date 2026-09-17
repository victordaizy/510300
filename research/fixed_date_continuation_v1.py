"""按显式日期续接固定研究账户；共用原依赖图、费用、模型与交易引擎。"""
import argparse
from contextlib import contextmanager
import json
import msvcrt
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.adaptive_allocation_v1 import normalize_dividends, target_request, summarize
from research.post_selection_continuous_replay_v1 import Pipeline, ROOT, INPUTS, PRIMARY
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account, simulate_policy, simulate_rearmed_exit, unpack
from research.post_selection_continuous_factors_v1 import align_decisions
from research.simple_intraday_protection_v1 import make_rules as session_rules


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def path(value):
    return (ROOT / value).resolve()


def canonical(frame, template):
    require(not (set(frame.columns) - set(template.columns)), '新增账户出现未定义列')
    return frame.reindex(columns=template.columns).astype(template.dtypes.to_dict())


def dates_to_append(data, calendar, source_close, cutoff, completed_at, dividend_coverage):
    """只允许已完整收盘且与上一状态相接的官方交易日。"""
    source_close, cutoff = pd.Timestamp(source_close), pd.Timestamp(cutoff)
    calendar = pd.DatetimeIndex(calendar).sort_values()
    require(source_close < cutoff, '没有新增截止日')
    require(source_close in calendar and cutoff in calendar, '起点或截止日不是官方交易日')
    stamp = pd.Timestamp(completed_at)
    require(stamp.tzinfo is not None, '数据取得时间缺少时区')
    require(stamp >= cutoff.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15, minutes=5), '新日线尚未完整收盘')
    require(pd.Timestamp(dividend_coverage) >= cutoff, '新分红覆盖不足')
    actual = pd.DatetimeIndex(data.date)
    require(actual.is_monotonic_increasing and actual.is_unique, '输入日期未严格递增')
    expected = calendar[(calendar > source_close) & (calendar <= cutoff)]
    observed = actual[(actual > source_close) & (actual <= cutoff)]
    require(observed.equals(expected), '新增日期缺失或混入非交易日')
    require(len(expected) > 0 and actual[-1] == cutoff, '输入末日与截止日不同')
    future = calendar[calendar > cutoff]
    require(len(future) > 0, '官方日历缺少下一交易日')
    return expected, str(future[0].date())


@contextmanager
def exclusive_run(folder):
    """Windows内核锁随进程释放，残留文件不被当作仍有活进程。"""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a+b') as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ValueError('同一续算目录有进程持有运行锁') from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def check_monthly_records(data, dates, ridge, within):
    firsts = data.groupby(data.date.dt.to_period('M')).date.min()
    due = set(pd.DatetimeIndex(firsts[firsts.isin(dates)]).strftime('%Y-%m-%d'))
    for name, records in [('岭回归', ridge), ('持仓内模型', within)]:
        origins = [r['fit_origin'] for r in records]
        require(len(origins) == len(set(origins)) and origins == sorted(origins), '月度模型日期重复或无序')
        require(due.issubset(origins), '缺少原定月首记录：' + name)
        for r in records:
            if r['fit_origin'] in due:
                require(data.date.iloc[int(r['fit_index'])] == pd.Timestamp(r['fit_origin']), '月首模型索引与输入日期不一致')
                require(r.get('latest_exit_index', r['fit_index']) <= r['fit_index'], '模型使用未成熟退出')


class DatePipeline(Pipeline):
    def __init__(self, settings):
        self.request = settings
        self.cfg = read(path(settings['configuration']))
        self.source, self.output = path(settings['source_folder']), path(settings['output_folder'])
        require(self.source != self.output and self.source not in self.output.parents and self.output not in self.source.parents,
                '新增输出必须与旧账户目录分开')
        self.cutoff = settings['cutoff']
        self.data = pd.read_parquet(path(settings['features']))
        self.data = self.data[self.data.date.le(self.cutoff)].reset_index(drop=True)
        self.div = normalize_dividends(pd.read_csv(path(settings['dividends'])))
        self.graph = {n['node']: n for n in read(INPUTS / 'dependency_graph.json')['nodes']}
        self.first = int(np.flatnonzero(self.data.date.ge(self.cfg['evaluation_start']))[0])
        self.accounts, self.targets, self.settings = {}, {}, {}
        self.checks, self.target_checks, self.calls = [], [], {}
        self.delta_receipts, self.recording_rows = [], []
        self.source_targets = pd.read_parquet(self.source / 'factors/all_required_targets.parquet')
        self.source_close = pd.Timestamp(self.source_targets.date.iloc[-1])
        old_features = pd.read_parquet(path(settings['source_features']))
        old_features = old_features[old_features.date.le(self.source_close)].reset_index(drop=True)
        pd.testing.assert_frame_equal(self.data.iloc[:len(old_features)].reset_index(drop=True), old_features, check_exact=True)
        self.ridge = read(path(settings['ridge_models']))['models']
        self.within = read(path(settings['within_models']))['models']
        if isinstance(self.ridge, dict):
            self.ridge = self.ridge['D60_INTRA__RIDGE']
        receipt = read(path(settings['admission_receipt']))
        require(receipt['accepted'] is True and receipt['cutoff'] == self.cutoff, '输入接纳回执不匹配')
        require(receipt['features_sha256'] == digest(path(settings['features'])) and
                receipt['dividends_sha256'] == digest(path(settings['dividends'])), '输入文件与接纳记录不同')
        require(pd.Timestamp(receipt['completed_at']) <= pd.Timestamp(now()), '输入回执使用未来时间')
        self.mode = settings['mode']
        require(self.mode in {'HISTORICAL_ENGINE_EQUIVALENCE', 'FIXED_RESEARCH_CONTINUATION'}, '未定义的续算类型')
        if self.mode == 'FIXED_RESEARCH_CONTINUATION':
            require(receipt['evidence_class'] == 'NEW_COMPLETE_DAILY_INPUTS', '未来研究必须使用新日线接纳记录')
            require(pd.Timestamp(self.cutoff) > pd.Timestamp('2026-09-11'), '旧历史不能登记为冻结后新数据')
            origin = read(path(settings['research_origin']))
            require(pd.Timestamp(origin['recorded_at']) < pd.Timestamp('2026-09-14T09:30:00+08:00'), '缺少开盘前研究起点')
        calendar = pd.to_datetime(pd.read_csv(ROOT / 'data/reference/sse_trade_calendar_2026.csv').trade_date)
        self.new_dates, self.next_date = dates_to_append(self.data, calendar, self.source_close, self.cutoff,
            receipt['completed_at'], receipt['dividend_coverage_through'])
        check_monthly_records(self.data, self.new_dates, self.ridge, self.within)
        self.session = session_rules(self.data)['D60_INTRA']

    def target(self, key, cost, values, check=True):
        values = np.asarray(values, float)
        require(len(values) == len(self.data) and (np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(),
                '来源目标超出定义：' + key)
        column = key + '__' + cost
        if column in self.source_targets:
            previous = self.source_targets[column].to_numpy(float)
            first = int(np.flatnonzero(self.data.date.ge(self.graph[key]['replay_start']))[0]) - 1
            np.testing.assert_allclose(values[first:len(previous)], previous[first:], rtol=1e-12, atol=1e-12, equal_nan=True,
                err_msg='已使用历史目标改变：' + column)
        self.targets[(key, cost)] = values
        return values

    def save_factors(self, name, frame):
        folder = self.output / 'factors'
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / (name + '.parquet')
        if output.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(output), frame, check_exact=True)
        else:
            frame.to_parquet(output, index=False)

    def account(self, key, cost, kind='target', values=None, rule=None, spec=None, controller_factory=None, request=None):
        require((key, cost) not in self.accounts, '共享账户重复计算')
        cfg = self.setting(key)
        source, folder = self.source / 'accounts' / cost / key, self.output / 'accounts' / cost / key
        old = pd.read_parquet(source / 'ledger.parquet')
        old_decisions = pd.read_parquet(source / 'decisions.parquet')
        snapshot = read(source / 'checkpoint.json')
        previous = unpack(snapshot['state'])
        require(previous['asof_date'] == self.source_close and old.date.iloc[-1] == self.source_close, '来源断点不同')
        require(old_decisions.execution_date.iloc[-1] == self.new_dates[0], '已保存申请未对应首个新增开盘')
        cached = (folder / 'checkpoint.json').exists()
        if cached:
            ledger, decisions = pd.read_parquet(folder / 'ledger.parquet'), pd.read_parquet(folder / 'decisions.parquet')
            delta, dd = pd.read_parquet(folder / 'delta_ledger.parquet'), pd.read_parquet(folder / 'delta_decisions.parquet')
            state = read(folder / 'checkpoint.json')
            recording = read(folder / 'recording.json')
        else:
            args = (self.data, self.div, cfg, cfg['costs'][cost], self.graph[key]['replay_start'])
            kwargs = {'next_execution_date': self.next_date, 'resume': snapshot}
            if kind == 'target':
                fn = simulate_indexed_request_account
                args += (key,)
                kwargs.update(targets=values, event_mask=np.ones(len(self.data), bool),
                    request_policy=request or (lambda a, p, v, c, m, t: target_request(a, p, v, c)))
            else:
                fn = simulate_rearmed_exit if kind == 'rearmed' else simulate_policy
                args += (rule, spec)
                if kind == 'rearmed':
                    kwargs['controller'] = controller_factory()
            run = fn(*args, **kwargs)
            delta, dd, state = run[0], run[1], run[-1]
            dd['decision_time'] = pd.to_datetime(dd.origin) + pd.Timedelta(hours=15, minutes=5)
            dd['source_model'], dd['source_cost'] = key, cost
            delta, dd = canonical(delta, old), canonical(dd, old_decisions)
            ledger, decisions = pd.concat([old, delta], ignore_index=True), pd.concat([old_decisions, dd], ignore_index=True)
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / 'ledger.parquet', index=False)
            decisions.to_parquet(folder / 'decisions.parquet', index=False)
            delta.to_parquet(folder / 'delta_ledger.parquet', index=False)
            dd.to_parquet(folder / 'delta_decisions.parquet', index=False)
            if kind != 'target':
                cycles = pd.read_csv(source / 'cycles.csv')
                cycles = pd.concat([cycles[cycles.exit_date.notna()], run[2]], ignore_index=True)
                require(not cycles.cycle_id.duplicated().any(), '持仓周期重复')
                cycles.to_csv(folder / 'cycles.csv', index=False, encoding='utf-8-sig')
            recording = {'actual_recorded_at': now(), 'simulation_clock_note': '决定表15:05字段为规则模拟时钟；实际保存时间单列',
                'mode': self.mode, 'cutoff': self.cutoff}
            write_json(folder / 'recording.json', recording)
            write_json(folder / 'checkpoint.json', state, exclusive=True)
        require(pd.DatetimeIndex(delta.date).equals(self.new_dates), '实际新增账户日期不同')
        require(delta.requested_quantity.iloc[0] == previous['pending']['requested_quantity'], '首个申请没有从旧状态继承')
        pd.testing.assert_frame_equal(ledger.iloc[:len(old)].reset_index(drop=True), old, check_exact=True)
        pd.testing.assert_frame_equal(decisions.iloc[:len(old_decisions)].reset_index(drop=True), old_decisions, check_exact=True)
        pd.testing.assert_frame_equal(ledger.iloc[len(old):].reset_index(drop=True), delta, check_exact=True)
        pd.testing.assert_frame_equal(decisions.iloc[len(old_decisions):].reset_index(drop=True), dd, check_exact=True)
        if kind == 'target':
            np.testing.assert_allclose(decisions.reference_weight, np.asarray(values)[decisions.origin_index], rtol=1e-12, atol=1e-12, equal_nan=True)
        final = unpack(state['state'])
        require(final['asof_date'] == pd.Timestamp(self.cutoff) and final['account']['shares'] == ledger.shares.iloc[-1], '新增状态与末日不一致')
        require(ledger.accounting_error.abs().max() < 1e-6 and ledger.mark_clock.eq('CLOSE').all(), '账户财富或结算时钟错误')
        require(len(decisions) == len(ledger) + 1 and decisions.execution_date.iloc[-1] == pd.Timestamp(self.next_date), '新增决定日期不完整')
        self.accounts[(key, cost)] = (ledger, decisions, state)
        self.target(key, cost, align_decisions(self.data, decisions), check=False)
        self.delta_receipts.append({'node': key, 'cost': cost, 'source_close': str(self.source_close.date()),
            'source_checkpoint_sha256': digest(source / 'checkpoint.json'), 'old_rows_reused': len(old), 'incremental_rows': len(delta),
            'start_open': str(self.new_dates[0].date()), 'end_close': self.cutoff, 'cached': cached,
            'actual_recorded_at': recording['actual_recorded_at'], 'old_prefix_exact': True})
        for row in dd.itertuples():
            before_next = pd.Timestamp(recording['actual_recorded_at']) < pd.Timestamp(row.execution_date).tz_localize('Asia/Shanghai') + pd.Timedelta(hours=9, minutes=30)
            self.recording_rows.append({'node': key, 'cost': cost, 'origin': str(row.origin.date()),
                'next_open_date': str(row.execution_date.date()), 'actual_recorded_at': recording['actual_recorded_at'],
                'saved_before_next_open': before_next, 'historical_engine_validation': self.mode == 'HISTORICAL_ENGINE_EQUIVALENCE'})
        print(f"{'复用' if cached else '增量'} {len(self.accounts)}/22：{key}／{cost}，{len(delta)}日。", flush=True)
        return ledger, decisions, state


def run(settings_path):
    settings_path = Path(settings_path).resolve()
    settings = read(settings_path)
    output = path(settings['output_folder'])
    binding = {'settings_sha256': digest(settings_path), 'engine_sha256': digest(Path(__file__)),
        'files': {key: digest(path(settings[key])) for key in ['configuration', 'features', 'source_features', 'dividends',
                   'ridge_models', 'within_models', 'admission_receipt']},
        'source_checkpoints': {str(p.relative_to(path(settings['source_folder']))): digest(p)
            for p in sorted(path(settings['source_folder']).glob('accounts/*/*/checkpoint.json'))}}
    require(len(binding['source_checkpoints']) == 22, '完整来源状态不足22条')
    with exclusive_run(output):
        start_path = output / 'RUN_STARTED.json'
        if start_path.exists():
            require(read(start_path)['binding'] == binding, '同一输出目录的参数或来源已经改变')
        else:
            write_json(start_path, {'started_at': now(), 'binding': binding}, exclusive=True)
        if (output / 'result.json').exists():
            stored = read(output / 'result.json')
            require(stored['status'] == 'COMPLETED_DATE_PARAMETERIZED_CONTINUATION', '保存结果未完成')
            print('同一日期已经完成，直接复用保存结果，未调用账户引擎。', flush=True)
            return stored
        began = time.perf_counter()
        pipeline = DatePipeline(settings).run()
        metrics = [{'model': key, 'cost': cost, 'name': '固定两来源连续账户', **summarize(value[0], pipeline.cfg)}
            for (key, cost), value in pipeline.accounts.items() if key == PRIMARY]
        pd.DataFrame(pipeline.delta_receipts).to_csv(output / 'incremental_receipts.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(pipeline.recording_rows).to_csv(output / 'decision_recording_times.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(metrics).to_csv(output / 'continuous_metrics.csv', index=False, encoding='utf-8-sig')
        result = {'completed_at': now(), 'status': 'COMPLETED_DATE_PARAMETERIZED_CONTINUATION', 'mode': pipeline.mode,
            'source_close': str(pipeline.source_close.date()), 'cutoff': pipeline.cutoff, 'next_official_trading_day': pipeline.next_date,
            'existing_accounts': len(pipeline.accounts), 'incremental_trading_days': len(pipeline.new_dates),
            'old_account_rows_reused': sum(r['old_rows_reused'] for r in pipeline.delta_receipts),
            'incremental_account_rows': sum(r['incremental_rows'] for r in pipeline.delta_receipts),
            'cached_accounts': sum(r['cached'] for r in pipeline.delta_receipts), 'all_metrics': metrics,
            'core_seconds': time.perf_counter() - began, 'new_model_fits': 0, 'new_candidates': 0,
            'goal_achieved': False, 'independent_performance_validation': False, 'position_impact': 0}
        write_json(output / 'result.json', result, exclusive=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用已接纳新输入，只从保存状态续算固定研究账户。')
    parser.add_argument('--settings', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().settings), ensure_ascii=False), flush=True)
