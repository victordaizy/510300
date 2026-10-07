# T11公告时钟与更正候选准备包

本包完成公告元数据的日期映射和更正候选目录，尚未计算O02价格反应、T11交易或账户夏普。夏普1.2目标未达成。这是本地准备结果，不是外部审阅或独立前向验证。

依次阅读：研究结论.md、用户请求与边界.md、evidence/clock/result.json、evidence/corrections/result.json、evidence/corrections/title_status_clock_addendum.json、审阅提示词.md。

FILE_INDEX.csv是本包唯一完整文件索引；两个evidence目录分别保留原冻结记录。标题状态时钟的追加说明是冻结后的独立追加证据，原冻结记录未被修改。

独立复算只需Python、pandas、numpy：在解压目录运行 `python code/verify_factor96_t11_metadata_packet_v1.py .`。该过程只读保存资料，不联网、不产生账户。提供的原生产脚本保留仓库接口以供代码检查；独立复算脚本不依赖该仓库。
