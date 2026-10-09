# 最终交付身份与复现

本页对应RESULTS.md的实际192个原合同候选账户及120个风险诊断/对照。原合同候选通过0/192；改变仓位/风控的5个单场景数字通过不计作原目标成功，且没有任何风险版本同时通过原四场景。goal_achieved=false，independent_validation=NOT_ESTABLISHED，overfitting_removed=false，orders_authorized=false。

## 最终会话包

- 文件：510300_macro_increment_risk_20261009.zip
- 字节数：24965891
- SHA-256：ccbcc1af9d588a924e966f8c22a7792092eb57754d8cb7d95c072f24f191b907
- 内容文件470个，另MANIFEST.json一份；内容总71688985字节。
- 所有文件字节数/SHA-256核验以及ZIP逐项CRC核验通过。
- 包含312个完整账户daily/orders/cycles、模型/训练时钟、必要原行情/股息/风险/父信号、信用原件及CSV、预测、区间与失败记录、代码。
- GitHub只保存方法登记、信用追加登记、来源恢复工作流和报告；完整新代码与账本在上述会话交付包，不冒称已提交整个Git树。

独立文件：510300_macro_increment_192_metrics.csv；510300_risk_diagnostic_120_metrics.csv；510300_macro_increment_risk_report_20261009.md。

## 实際复核

在与原工作目录分开的交付目录执行：python macro_residual_v2/replay.py --output verification_runs/check_01 --refit，得到REPLAY_PASSED 312 ACCOUNTS 4 FORECAST_CHECKS。全部日账/订单/周期数组逐值一致；四套重新拟合的预测以1e-12容差通过。该核验额外2595次模型拟合，不计新候选或独立市场证据。

回执和日志放入delivery_verification/，因此交付后示例verification_runs/check_01路径仍可用于首次复跑。后续必须用新的输出路径。最终打包仅追加报告说明与核验记录、清理可重建编译缓存、更新MANIFEST，金融代码和原结果不变。先完成独立目录重放，再完成最终ZIP的清单/CRC核验；没有冒称最终ZIP解压后又重拟合一次。

默认复核：python macro_residual_v2/replay.py --output verification_runs/check_01。
完整预测重拟合：追加--refit。
依赖requirements.txt；代码不联网下载数据、不接券商、不下单。原A使用保存父信号，未部署未来日期自动生成系统。

## 信用源续接

GitHub Actions run37932491488，artifact11616278513，名称macro-credit-source-20261009，原附件146690字节，SHA-256 f9cf4d0d141e57ebebd914cd404a205624de703f2e8a210d9dbb64663ad363b4，原保留期限2026-11-08。专用download_workflow_artifact可取二进制；fetch不可用于ZIP。到期时按.github/workflows/macro-credit-source-20261009.yml与catalog只还原必要四件。来源仍仅PASS_DISCOVERY_ONLY，1006条TLS未验证、无独立供应商或历史首版认证；字节核验不消除这些限制。

原合同未改，100%执行仅为独立标记诊断。D021992的100%后段年化14.4612%、夏普1.6748、回撤5.0550%，早期年化2.4338%、回撤12.3390%且停止，不能宣传为稳定达标。全部研究历史已见；不以本轮回测或复核次数代替独立市场信息。