# 文件与附件索引

- [快照摘要](snapshot.json)：数量、总字节数、源 Git 提交、版本和检查范围。
- [Release 附件](assets.json)：每个附件的下载地址、大小、类型和 SHA-256。
- [研究目录](studies.csv)：研究目录、文件数、总大小和报告入口。
- [目录规模](directory_summary.json)：按原项目一级目录汇总。
- [排除说明](exclusions.json)：范围外材料及运行目录的排除原因。
- [本机运行环境](runtime.json)：Python 及包版本记录。
- [附件内容回读核验](SAVED_ASSETS_VERIFICATION.json)：全部188,994个Release原文件的CRC、字节数及SHA-256验证。
- [还原路径忽略规则验证](IGNORE_RULES_VERIFICATION.json)：还原资料与主仓库文件的Git规则覆盖。
- [研究汇总引用验证](STATUS_REFERENCES_VERIFICATION.json)：本快照汇总文件的114个引用均已纳入。
- [来源Git规则原件](source-control/README.md)：来源项目忽略和换行规则的原始副本。

## 原始文件清单

每份 CSV 最多 10,000 行，按路径排序；CSV 文件自身的 SHA-256 由 `snapshot.json` 固定。

- [files-001.csv](files-001.csv)：10,000 个源文件。
- [files-002.csv](files-002.csv)：10,000 个源文件。
- [files-003.csv](files-003.csv)：10,000 个源文件。
- [files-004.csv](files-004.csv)：10,000 个源文件。
- [files-005.csv](files-005.csv)：10,000 个源文件。
- [files-006.csv](files-006.csv)：10,000 个源文件。
- [files-007.csv](files-007.csv)：10,000 个源文件。
- [files-008.csv](files-008.csv)：10,000 个源文件。
- [files-009.csv](files-009.csv)：10,000 个源文件。
- [files-010.csv](files-010.csv)：10,000 个源文件。
- [files-011.csv](files-011.csv)：10,000 个源文件。
- [files-012.csv](files-012.csv)：10,000 个源文件。
- [files-013.csv](files-013.csv)：10,000 个源文件。
- [files-014.csv](files-014.csv)：10,000 个源文件。
- [files-015.csv](files-015.csv)：10,000 个源文件。
- [files-016.csv](files-016.csv)：10,000 个源文件。
- [files-017.csv](files-017.csv)：10,000 个源文件。
- [files-018.csv](files-018.csv)：10,000 个源文件。
- [files-019.csv](files-019.csv)：10,000 个源文件。
- [files-020.csv](files-020.csv)：6,577 个源文件。

## 字段定义

| 字段 | 含义 |
|---|---|
| `path` | 相对原项目根目录的完整原始路径 |
| `bytes` | 完整原文件的字节数 |
| `sha256` | 完整原文件的 SHA-256 |
| `storage` | `git` 或 `release` |
| `assets` | 还原该文件所需附件名；原包分片按顺序排列 |
| `reason` | 收录原因 |
| `captured_mtime_ns` | 原文件在实际读取时的纳秒级修改时间戳 |
| `changed_since_inventory` | 盘点与实际读取之间大小或修改时间是否发生变化 |
