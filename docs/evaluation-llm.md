# Agent 评测报告

- 检索模式：neo4j
- Planner：LLMRecipePlanner（失败时规则降级）
- 场景数量：54
- 任务成功率：100.0%
- 检索命中率：100.0%
- 相关性命中率：100.0%
- 约束满足率：100.0%
- 引用覆盖率：100.0%
- 平均延迟：5.69 秒
- Planner 调用次数：165
- LLM 直接规划比例：100.0%
- 规则降级比例：0.0%

| 场景 | 成功 | 检索命中 | 相关性 | 安全约束 | 引用覆盖 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| recipe-001 | True | True | True | True | 100% | 7.40s |
| recipe-002 | True | True | True | True | 100% | 4.11s |
| recipe-003 | True | True | True | True | 100% | 3.85s |
| recipe-004 | True | True | True | True | 100% | 4.98s |
| recipe-005 | True | True | True | True | 100% | 5.95s |
| recipe-006 | True | True | True | True | 100% | 7.33s |
| recipe-007 | True | True | True | True | 100% | 10.39s |
| recipe-008 | True | True | True | True | 100% | 4.18s |
| recipe-009 | True | True | True | True | 100% | 4.31s |
| recipe-010 | True | True | True | True | 100% | 4.29s |
| safe-001 | True | True | True | True | 100% | 5.71s |
| safe-002 | True | True | True | True | 100% | 9.14s |
| menu-001 | True | True | True | True | 100% | 15.46s |
| menu-002 | True | True | True | True | 100% | 22.15s |
| recipe-015 | True | True | True | True | 100% | 4.26s |
| recipe-016 | True | True | True | True | 100% | 4.58s |
| recipe-017 | True | True | True | True | 100% | 7.46s |
| recipe-018 | True | True | True | True | 100% | 5.12s |
| recipe-019 | True | True | True | True | 100% | 3.44s |
| recipe-020 | True | True | True | True | 100% | 4.78s |
| recipe-021 | True | True | True | True | 100% | 4.27s |
| recipe-022 | True | True | True | True | 100% | 5.48s |
| recipe-023 | True | True | True | True | 100% | 5.10s |
| recipe-024 | True | True | True | True | 100% | 5.23s |
| recipe-025 | True | True | True | True | 100% | 5.49s |
| recipe-026 | True | True | True | True | 100% | 5.65s |
| recipe-027 | True | True | True | True | 100% | 5.22s |
| recipe-028 | True | True | True | True | 100% | 4.16s |
| recipe-029 | True | True | True | True | 100% | 6.30s |
| recipe-030 | True | True | True | True | 100% | 4.98s |
| recipe-031 | True | True | True | True | 100% | 5.73s |
| recipe-032 | True | True | True | True | 100% | 7.41s |
| recipe-033 | True | True | True | True | 100% | 4.44s |
| recipe-034 | True | True | True | True | 100% | 4.46s |
| recipe-035 | True | True | True | True | 100% | 7.59s |
| recipe-036 | True | True | True | True | 100% | 4.85s |
| recipe-037 | True | True | True | True | 100% | 3.84s |
| recipe-038 | True | True | True | True | 100% | 5.71s |
| recipe-039 | True | True | True | True | 100% | 3.89s |
| recipe-040 | True | True | True | True | 100% | 3.41s |
| recipe-041 | True | True | True | True | 100% | 4.18s |
| recipe-042 | True | True | True | True | 100% | 5.87s |
| recipe-043 | True | True | True | True | 100% | 5.77s |
| recipe-044 | True | True | True | True | 100% | 3.48s |
| recipe-045 | True | True | True | True | 100% | 4.51s |
| recipe-046 | True | True | True | True | 100% | 4.45s |
| recipe-047 | True | True | True | True | 100% | 4.05s |
| recipe-048 | True | True | True | True | 100% | 4.48s |
| recipe-049 | True | True | True | True | 100% | 4.83s |
| recipe-050 | True | True | True | True | 100% | 4.46s |
| recipe-051 | True | True | True | True | 100% | 4.42s |
| recipe-052 | True | True | True | True | 100% | 4.28s |
| recipe-053 | True | True | True | True | 100% | 5.53s |
| recipe-054 | True | True | True | True | 100% | 4.82s |
