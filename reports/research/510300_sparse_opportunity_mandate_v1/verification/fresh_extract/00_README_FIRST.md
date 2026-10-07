# 510300：20 万元、夏普 1.5、回撤 10%、稀疏机会

本轮完成最新研究约束和既有账户筛查，目标尚未达成。读者可直接检查本包保存的原始指标、原始关闭裁决与本轮去重结果；不需要访问用户电脑即可复算本轮筛查。

阅读顺序：

1. `研究结论.md`：结论、六个旧候选的跨时期表现、下一项研究。
2. `用户需求.md` 和 `source_snapshot/config/510300_sparse_opportunity_mandate_v1.json`：用户原话与本轮约束。
3. `deduplicated_candidate_comparison.csv`：九个曾有单档成本数值达线的模型，包含六个双成本达线模型。
4. `all_saved_metric_rows.csv`、`metric_source_inventory.csv`：214 份指标、2,710 条含重复对照的完整记录。
5. `passing_model_saved_cycles.csv`：这些模型的压力成本持仓周期。名称中的 passing 仅指至少一档成本的历史指标达线，不表示策略验收通过。
6. `source_snapshot/`：原研究的保存指标、配置、结果与裁决；较早时期对照、原周期表及规则可定位到对应研究目录。
7. `GPT审阅提示词.md`：可直接复制的审阅请求。

复算入口为 `source_snapshot/scripts/review_510300_sparse_opportunity_v1.py`，只依赖 Python 标准库。它读取本包内已有指标和周期，不训练模型、不调用行情接口、不新建回测账户。新输出写入这个脚本所在快照根目录下的 `reports/research/510300_sparse_opportunity_mandate_v1`。

本包不是旧策略全流程复现包：未收录全部市场原始行情、所有旧账户逐日账簿和父模型训练数据。本轮没有据此宣称旧账户计算全部正确，也没有进行外部 GPT 审阅。具体排除项见 `EXCLUSIONS.md`。
