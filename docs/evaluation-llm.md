# Agent 评测报告

- 检索模式：neo4j
- Planner：LLMRecipePlanner（失败时规则降级）
- 场景数量：54
- 任务成功率：98.1%
- 检索命中率：100.0%
- 相关性命中率：100.0%
- 约束满足率：100.0%
- 引用覆盖率：100.0%
- 平均延迟：6.27 秒
- Planner 调用次数：160
- LLM 直接规划比例：100.0%
- 规则降级比例：0.0%

| 场景 | 成功 | 检索命中 | 相关性 | 安全约束 | 引用覆盖 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| recipe-001 | True | True | True | True | 100% | 12.97s |
| recipe-002 | True | True | True | True | 100% | 5.13s |
| recipe-003 | True | True | True | True | 100% | 6.16s |
| recipe-004 | True | True | True | True | 100% | 4.73s |
| recipe-005 | True | True | True | True | 100% | 7.23s |
| recipe-006 | True | True | True | True | 100% | 8.12s |
| recipe-007 | True | True | True | True | 100% | 4.47s |
| recipe-008 | True | True | True | True | 100% | 5.20s |
| recipe-009 | True | True | True | True | 100% | 12.61s |
| recipe-010 | True | True | True | True | 100% | 3.85s |
| safe-001 | True | True | True | True | 100% | 5.80s |
| safe-002 | True | True | True | True | 100% | 5.53s |
| menu-001 | True | True | True | True | 100% | 17.74s |
| menu-002 | False | True | True | True | 100% | 24.18s |
| recipe-015 | True | True | True | True | 100% | 15.38s |
| recipe-016 | True | True | True | True | 100% | 7.13s |
| recipe-017 | True | True | True | True | 100% | 4.15s |
| recipe-018 | True | True | True | True | 100% | 5.56s |
| recipe-019 | True | True | True | True | 100% | 4.03s |
| recipe-020 | True | True | True | True | 100% | 4.02s |
| recipe-021 | True | True | True | True | 100% | 4.20s |
| recipe-022 | True | True | True | True | 100% | 3.44s |
| recipe-023 | True | True | True | True | 100% | 5.38s |
| recipe-024 | True | True | True | True | 100% | 5.66s |
| recipe-025 | True | True | True | True | 100% | 5.66s |
| recipe-026 | True | True | True | True | 100% | 4.71s |
| recipe-027 | True | True | True | True | 100% | 4.60s |
| recipe-028 | True | True | True | True | 100% | 3.56s |
| recipe-029 | True | True | True | True | 100% | 5.22s |
| recipe-030 | True | True | True | True | 100% | 4.90s |
| recipe-031 | True | True | True | True | 100% | 4.25s |
| recipe-032 | True | True | True | True | 100% | 6.54s |
| recipe-033 | True | True | True | True | 100% | 10.31s |
| recipe-034 | True | True | True | True | 100% | 5.05s |
| recipe-035 | True | True | True | True | 100% | 4.11s |
| recipe-036 | True | True | True | True | 100% | 5.44s |
| recipe-037 | True | True | True | True | 100% | 5.79s |
| recipe-038 | True | True | True | True | 100% | 5.86s |
| recipe-039 | True | True | True | True | 100% | 4.18s |
| recipe-040 | True | True | True | True | 100% | 7.75s |
| recipe-041 | True | True | True | True | 100% | 6.04s |
| recipe-042 | True | True | True | True | 100% | 4.93s |
| recipe-043 | True | True | True | True | 100% | 4.55s |
| recipe-044 | True | True | True | True | 100% | 3.98s |
| recipe-045 | True | True | True | True | 100% | 5.18s |
| recipe-046 | True | True | True | True | 100% | 4.18s |
| recipe-047 | True | True | True | True | 100% | 4.42s |
| recipe-048 | True | True | True | True | 100% | 6.96s |
| recipe-049 | True | True | True | True | 100% | 5.38s |
| recipe-050 | True | True | True | True | 100% | 5.01s |
| recipe-051 | True | True | True | True | 100% | 4.26s |
| recipe-052 | True | True | True | True | 100% | 4.29s |
| recipe-053 | True | True | True | True | 100% | 4.05s |
| recipe-054 | True | True | True | True | 100% | 4.89s |
