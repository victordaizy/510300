# 510300 PCF/IOPV前瞻运行审计（2026-08-19）

- 研究状态：`COLLECTING_NOT_ELIGIBLE`
- 观测交易日：4
- 完整质量日：2
- 最近有效交易日：`2026-08-18`
- 2026-08-19任务状态：`FAILED_SUPPLIER_TIMEOUT`
- 失败主机：`query.sse.com.cn`
- 精确错误：`ReadTimeout: HTTPSConnectionPool(host='query.sse.com.cn', port=443): Read timed out. (read timeout=15.0)`
- 旧计划任务返回码：0，但该返回码不可信；旧后台启动器在采集子进程结束前已经返回。
- 失败日志SHA-256：`c0e7f96481e01e716cc5620568de29dcbf04442ab4e146157a64ce0943d9ef9b`
- 未执行历史IOPV回填。

修复后，计划任务改为前台运行并传播真实退出码，单次运行允许连续5次采集失败后终止，Windows任务最多重启3次。任务固定使用研究采集模式，跳过R5刷新和小账户纸面信号。

最低质量审计要求20个完整日，特征冻结要求40日，首次未见评价要求80日，复制评价要求120日。当前均未达到。

仓位映射、订单生成、券商连接和实盘全部关闭。

