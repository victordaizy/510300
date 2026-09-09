# 仓库目录导航

| 目录或文件 | 内容 | 主要存放位置 |
|---|---|---|
| `config/` | 研究参数、冻结协议、权限与清单 | Git |
| `research/` | 各项研究实现 | Git |
| `scripts/` | 采集、分析、运行和历史交付入口 | Git |
| `src/`、`backtest/`、`market_data/` | 公共模块与既有依赖 | Git |
| `tests/` | 原研究测试与新增还原工具测试 | Git |
| `docs/` | 研究说明、决策记录与仓库维护指南 | Git |
| `reports/`、`paper/` | 历史报告、结果、回执和证据 | 可读报告在 Git；批量证据在 Release |
| `data/` | 行情、原始来源、特征、冻结数据和账本 | Release，按原路径还原 |
| `deliverables/`、`review_packages/` | 历史审阅 ZIP、展开材料和交付说明 | Release，按原路径还原 |
| `catalog/` | 全部源文件、附件和研究目录索引 | Git |
| `tools/repository/` | 标准库下载还原与摘要验证工具 | Git |

## 阅读顺序

1. 阅读根目录 `README.md`，确认快照版本和文件范围。
2. 通过 `catalog/studies.csv` 找到具体研究目录及报告入口；根目录 `RESEARCH_STATUS.md` 保留已有状态导航。
3. 按具体报告阅读协议、权限、来源、结果与反证；同一研究存在多个版本时，逐一核对编号与时间，不把旧摘要当成新结果。
4. 如果报告引用的文件通过 Release 保存，使用还原工具按文件或目录前缀下载。
5. 需要完整复核时，下载全部附件，再执行 `verify --all`。

## 文件索引的读法

`catalog/files-*.csv` 中的 `path` 是相对原项目根目录的路径；`storage=git` 表示原文件直接在主仓库中，`storage=release` 表示使用 `assets` 列所列附件还原。`sha256` 及 `bytes` 描述完整原文件，不是其分片；附件摘要另见 `catalog/assets.json`。

已有文档中可能包含来源电脑的绝对路径或当时的外部链接。它们属于冻结内容，因此没有批量改写。查阅时可根据路径中 `New project 8` 后的相对路径在文件索引中定位；还原工具将材料写到当前仓库根目录。

## 原始大包

大于单个附件容量目标的原包被切为 `.bin` 分片。分片不是独立 ZIP，不应分别解压。还原工具按索引顺序拼接，并验证原始 ZIP 的 SHA-256，然后恢复原来的中文或英文文件名。

## 当前检查的实际范围

CI 检查所有原始 Git 文件的大小和摘要、附件映射结构，以及还原工具的关键行为。它不下载全部历史数据、不运行研究模型，也不把结构检查称为安全审计、科学有效性结论或外部 GPT 审阅。

本次还独立回读了全部188,994个通过Release保存的原文件，核对ZIP CRC、成员集合、大小及SHA-256，并验证原包分片的顺序拼接摘要。结果见 `catalog/SAVED_ASSETS_VERIFICATION.json`。

还原后的Release材料由 `.gitignore` 中依据索引生成的规则管理。188,994条Release路径全部受到忽略规则覆盖，7,583条原始Git路径没有被误忽略。后续更新发布索引后，可运行 `python tools/repository/update_ignore_rules.py` 重新生成并用Git检查这些规则。
