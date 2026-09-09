# 零付费研究计划完成度审计 V1

- 生成时间：2026-08-26T22:56:52.080104+08:00
- 总状态：`IN_PROGRESS_PENDING_REQUIRED_EVIDENCE`
- 完成已被证明：`FALSE`
- 研究结论：`NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET`
- 当前决定：`PAUSE_AND_FIX_FOUNDATION`
- 已满足：18 / 26
- 待取得证据：8
- 矛盾或缺失：0

| ID | 要求 | 状态 | 当前证据摘要 | 下一步 |
|---|---|---|---|---|
| R01 | 两份用户文本均完整读取且第二份冲突时优先 | `PASS` | {"attachment_count":2,"precedence":"pasted-text-2"} | — |
| R02 | 总体裁决保持无已验证可交易 Alpha/强 Beta，并暂停修复基础设施 | `PASS` | {"overall_research_status":"NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET","decision":"PAUSE_AND_FIX_FOUNDATION"} | — |
| R03 | 至少 180 天零付费数据合同 | `PASS` | {"budget_cny":0,"lock_days":180,"registry_rows":16,"nonzero_cost_rows":0,"policy_status":"PASS","policy_sha256":"534bce83d18f78cc4893bbb861caa3826860d2c5a2e21a8143a4064d6858668b... | — |
| R04 | 活动研究流最多两条 | `PASS` | {"maximum":2,"active_ids":["PRIMARY_MARKET_PCF_IOPV","INDUSTRY_EXPECTATION_GAP"]} | — |
| R05 | 免费数据源注册表完整覆盖字段、时间、点时性、原始归档、许可和失败合同 | `PASS` | {"rows":16,"unique_source_ids":16,"registry_status":"PASS","missing_columns":[],"empty_required_cells":[],"nonzero_cost_source_ids":[],"raw_preservation_not_required":[],"canoni... | — |
| R06 | 主前瞻入口、配置和终局证据受冻结清单约束 | `PASS` | {"status":"PASS","manifest_sha256":"ae8215c5b7e35837564d464eb975054a2d79e2dd33a0cc25636b2c667fdcd1c3","content_sha256":"a3af06ae102c59d40b0105d94611a5d001f214d0edf1632f21a9c1772... | — |
| R07 | PCF/IOPV V1.2 通过冻结入口在合法窗口产生真实当日回执 | `PENDING_NEXT_WINDOW` | {"trading_day":true,"runtime_contract":{"runtime_version":"V1_7","task_receipt_directory":"reports/data_quality/primary_market_task_runs_v1_2","codex_receipt_directory":"reports... | 保留本日失败状态且不重试；等待下一交易日真实窗口 |
| R08 | PCF/IOPV 只按真实完整质量日推进 20/40/80/120 门槛 | `PENDING_MATURITY` | {"status":"COLLECTING_NOT_ELIGIBLE","observed_trading_days":4,"legacy_full_coverage_days":2,"v1_2_crosscheck_passed_days":0,"gates":[20,40,80,120],"evaluation_eligible":false,"c... | 继续严格前瞻；未达到相应门槛不得评价 |
| R25 | PCF修复后首20个交易机会达到内部程序95%和完整质量日90%验收门 | `PENDING_MATURITY` | {"status":"PENDING_MATURITY","effective_from":"2026-08-26","required_opportunities":20,"opportunity_count":1,"opportunity_dates":["2026-08-26"],"acceptance_window_complete":fals... | 继续累计首20个真实交易机会；缺失日计入分母且不得补跑 |
| R09 | 行业预期差使用内容寻址冻结快照且不消费共享 latest | `PASS` | {"audit_status":"PASS_FROZEN_INPUTS_CONTENT_ADDRESSED_AND_LATEST_ISOLATED","snapshot_id":"525b84ab6a65ee8d6fb5b4752852e12b4d09b78ffe506f56430c01e355da0d1d","shared_latest_consum... | — |
| R10 | 行业预期差按独立原点和非重叠 60 日块成熟，不提前评价 | `PENDING_MATURITY` | {"status":"COLLECTING_FORWARD","view_status":"NO_VIEW","origin_clusters":1,"mature_origin_clusters":0,"non_overlapping_60d_blocks":0,"calibration_gate":"20 origins and 4 blocks"... | 等待新官方原点和成熟结果；当前保持 NO_VIEW |
| R11 | T-only 成熟前公开输出仅含成熟度和完整性 | `PASS` | {"current_view":"NO_VIEW_UNTIL_FORWARD_MATURITY","forbidden_key_matches":[],"manifest":{"status":"PASS","manifest_sha256":"a390cdb2cbd8c1bfa3dc18dbd1055d2acf1792d39b337ff27bd08d... | — |
| R12 | T-only V1.2 计划任务产生首次真实本日回执 | `PENDING_CURRENT_RUN` | {"task_exact":true,"task":{"task_name":"Codex-510300-T-Only-Forward-V1","state":"Ready","next_run_time":"2026-08-27T16:30:00.0000000+08:00","last_run_time":"2026-08-26T22:38:58.... | 读取本日 V1.2 不可变回执；无回执时保持运行完整性未确认 |
| R13 | T-only 达到 252 个新交易日并闭合至少 3 个周期后才评价 | `PENDING_MATURITY` | {"new_trading_days":4,"closed_cycles":0,"required_new_trading_days":252,"required_closed_cycles":3,"current_view":"NO_VIEW_UNTIL_FORWARD_MATURITY","contract_recalculation":{"sta... | 继续自动追加账本，不查看中途表现 |
| R14 | 期权盘口先做零成本来源准入；未通过即关闭且不用代理 | `TERMINAL_EXPECTED` | {"qualification_status":"BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE","qualified_trial_days":0,"required_trial_days":20,"closure":{"active_supervisor_task":false,"scheduled_trial_enabl... | — |
| R15 | 现金选择权只消耗一次最终免费修复，数据门失败即终结且不读取收益 | `TERMINAL_EXPECTED` | {"status":"NO_VIEW_FREE_DATA_INSUFFICIENT","attempt_consumed":true,"event_count":21,"data_gate_pass_count":0,"price_values_read":false,"return_values_read":false,"second_repair_... | — |
| R16 | 当前 V1_8 Windows 任务在注销和重启后真实运行通过 | `PENDING_PERMISSION` | {"validation_status":"BLOCKED_PERMISSION_S4U_REGISTRATION_ACCESS_DENIED","current_task_runner":"\"C:\\Users\\戴周阳\\Documents\\New project 8\\scripts\\run_priority_forward_codex_a... | 需用户授权后对当前 V1_8 入口各完成一次注销与重启实测；此前保持未验证 |
| R17 | 研究、持仓映射、订单、券商连接和实盘授权保持分离 | `PASS` | {"research_only":true,"shadow_enabled":false,"position_mapping_enabled":false,"order_generation_enabled":false,"broker_connection_enabled":false,"live_trading_enabled":false} | — |
| R18 | Codex 09:20 与 19:00 自动化均为 ACTIVE 且固定 V1_8 清单哈希 | `PASS` | [{"id":"510300-pcf-iopv","status":"PASS","checks":{"active":true,"runner_pinned":true,"manifest_sha256_pinned":true,"schedule_matches":true},"path":"C:\\Users\\戴周阳\\.codex\\auto... | — |
| R26 | 实际调度面只保留授权的V1_8与T-only入口，期权、V3和旧版重复任务均停用 | `PASS` | {"status":"PASS","checks":{"windows_inventory_readable":true,"only_authorized_windows_tasks_enabled":true,"legacy_windows_tasks_disabled":true,"pcf_windows_task_exact":true,"t_o... | — |
| R19 | 外部免费源失败与内部程序失败使用不同状态和退出码 | `PASS` | {"external_status":"EXTERNAL_FREE_SOURCE_FAILED","program_status":"PROGRAM_FAILED","external_exit_code":3,"program_exit_code":1} | — |
| R20 | 失败、错过窗口和缺失数据不自动重试、不补跑、不回填 | `PASS` | {"pcf_restart_count":0,"pcf_start_when_available":false,"industry_historical_backfill":false,"option_historical_backfill":false,"t_only_backfill":false} | — |
| R21 | 哈希漂移的 V3 分支保持失败隔离，不重试也不复用旧信号 | `TERMINAL_EXPECTED` | {"daily_status":"FAILED","mismatch_count":3,"active_task":false,"latest_status_exists":false,"retry_or_repair_performed":false,"codex_automation_status":"PAUSED"} | — |
| R23 | 可转债官方权利条款与事前最大样本上限完成冻结裁定 | `PASS` | {"status":"CONTINUE_OFFICIAL_LIFECYCLE_EVIDENCE_ONLY","candidate_count":939,"official_right_terms_complete_count":854,"maximum_evaluable_sample_upper_bound_count":854,"maximum_e... | — |
| R24 | 可转债官方完整生命周期门：达到846才查双源，不足即NO_VIEW并停止 | `TERMINAL_EXPECTED` | {"status":"NO_VIEW_INSUFFICIENT_EVIDENCE","official_complete_lifecycle_count":845,"required_complete_lifecycle_count":846,"official_complete_lifecycle_fraction":0.89989350372736... | — |
| R22 | 本交易日收盘后由 V1_7 编排回执生成最新 V1.6 权威研究状态 | `PENDING_NEXT_WINDOW` | {"before_close_window":false,"acceptance_status":"FAILED_RUN","contract_valid":true,"successful":false,"industry_receipt_count":1,"industry_acceptable":false,"status_receipt_cou... | 保留本日失败回执且不补跑，等待下一交易日窗口 |

## 当前未完成项

- `R07`：PCF/IOPV V1.2 通过冻结入口在合法窗口产生真实当日回执；保留本日失败状态且不重试；等待下一交易日真实窗口
- `R08`：PCF/IOPV 只按真实完整质量日推进 20/40/80/120 门槛；继续严格前瞻；未达到相应门槛不得评价
- `R25`：PCF修复后首20个交易机会达到内部程序95%和完整质量日90%验收门；继续累计首20个真实交易机会；缺失日计入分母且不得补跑
- `R10`：行业预期差按独立原点和非重叠 60 日块成熟，不提前评价；等待新官方原点和成熟结果；当前保持 NO_VIEW
- `R12`：T-only V1.2 计划任务产生首次真实本日回执；读取本日 V1.2 不可变回执；无回执时保持运行完整性未确认
- `R13`：T-only 达到 252 个新交易日并闭合至少 3 个周期后才评价；继续自动追加账本，不查看中途表现
- `R16`：当前 V1_8 Windows 任务在注销和重启后真实运行通过；需用户授权后对当前 V1_8 入口各完成一次注销与重启实测；此前保持未验证
- `R22`：本交易日收盘后由 V1_7 编排回执生成最新 V1.6 权威研究状态；保留本日失败回执且不补跑，等待下一交易日窗口

## 矛盾或缺失

无。
