# Figure chart map

|   figure | question    | chart                              | data_scope                                       |
|---------:|:------------|:-----------------------------------|:-------------------------------------------------|
|        1 | 训练轨迹    | Trend / 3x3 small multiples        | 40 training batteries; policy facets             |
|        2 | 测试预测    | Uncertainty / 3x3 small multiples  | 9 test batteries; observed, prediction, interval |
|        3 | CV一致性    | Relationship / scatter             | 2,000 cutoff-150 OOF points                      |
|        4 | horizon误差 | Trend / multi-series line          | RMSE by horizons 1-50                            |
|        5 | 早期长度    | Trend / line with markers          | cutoff 50/75/100/125/150                         |
|        6 | Agent评审   | Comparison / ranked horizontal bar | six routes                                       |
|        7 | 特征重要性  | Ranking / horizontal bar           | outer-safe permutation importance                |
|        8 | 区间示例    | Uncertainty / line and band        | median-RMSE training battery                     |
|        9 | EOL敏感性   | Uncertainty / log-scale point-line | three degradation families                       |
|       10 | 同策略迁移  | Cohort / highlighted line          | one test battery and same-policy peers           |

- 字体：Microsoft YaHei
- 输出：240 dpi PNG，白底，静态论文图。
- 颜色不是唯一编码；关键系列同时使用线型、标记或直接标签。
