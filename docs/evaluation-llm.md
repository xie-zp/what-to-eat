# Agent 评测报告

- 检索模式：neo4j
- Planner：LLMRecipePlanner（失败时规则降级）
- 场景数量：54
- 任务成功率：100.0%
- 检索命中率：100.0%
- 相关性命中率：100.0%
- 约束满足率：100.0%
- 引用覆盖率：100.0%
- 平均延迟：8.38 秒
- Planner 调用次数：163
- LLM 直接规划比例：100.0%
- 规则降级比例：0.0%

| 场景 | 成功 | 检索命中 | 相关性 | 安全约束 | 引用覆盖 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| recipe-001 | True | True | True | True | 100% | 5.75s |
| recipe-002 | True | True | True | True | 100% | 7.14s |
| recipe-003 | True | True | True | True | 100% | 9.38s |
| recipe-004 | True | True | True | True | 100% | 23.56s |
| recipe-005 | True | True | True | True | 100% | 7.71s |
| recipe-006 | True | True | True | True | 100% | 14.06s |
| recipe-007 | True | True | True | True | 100% | 16.23s |
| recipe-008 | True | True | True | True | 100% | 11.81s |
| recipe-009 | True | True | True | True | 100% | 6.97s |
| recipe-010 | True | True | True | True | 100% | 7.15s |
| safe-001 | True | True | True | True | 100% | 12.06s |
| safe-002 | True | True | True | True | 100% | 9.68s |
| menu-001 | True | True | True | True | 100% | 29.21s |
| menu-002 | True | True | True | True | 100% | 34.15s |
| recipe-015 | True | True | True | True | 100% | 10.55s |
| recipe-016 | True | True | True | True | 100% | 7.77s |
| recipe-017 | True | True | True | True | 100% | 5.22s |
| recipe-018 | True | True | True | True | 100% | 4.21s |
| recipe-019 | True | True | True | True | 100% | 6.03s |
| recipe-020 | True | True | True | True | 100% | 4.66s |
| recipe-021 | True | True | True | True | 100% | 3.79s |
| recipe-022 | True | True | True | True | 100% | 11.88s |
| recipe-023 | True | True | True | True | 100% | 7.61s |
| recipe-024 | True | True | True | True | 100% | 9.13s |
| recipe-025 | True | True | True | True | 100% | 4.74s |
| recipe-026 | True | True | True | True | 100% | 7.30s |
| recipe-027 | True | True | True | True | 100% | 5.61s |
| recipe-028 | True | True | True | True | 100% | 5.82s |
| recipe-029 | True | True | True | True | 100% | 6.68s |
| recipe-030 | True | True | True | True | 100% | 6.47s |
| recipe-031 | True | True | True | True | 100% | 20.72s |
| recipe-032 | True | True | True | True | 100% | 5.82s |
| recipe-033 | True | True | True | True | 100% | 5.98s |
| recipe-034 | True | True | True | True | 100% | 5.15s |
| recipe-035 | True | True | True | True | 100% | 5.50s |
| recipe-036 | True | True | True | True | 100% | 5.55s |
| recipe-037 | True | True | True | True | 100% | 4.33s |
| recipe-038 | True | True | True | True | 100% | 10.18s |
| recipe-039 | True | True | True | True | 100% | 9.50s |
| recipe-040 | True | True | True | True | 100% | 3.70s |
| recipe-041 | True | True | True | True | 100% | 4.26s |
| recipe-042 | True | True | True | True | 100% | 4.93s |
| recipe-043 | True | True | True | True | 100% | 6.45s |
| recipe-044 | True | True | True | True | 100% | 3.88s |
| recipe-045 | True | True | True | True | 100% | 5.02s |
| recipe-046 | True | True | True | True | 100% | 5.38s |
| recipe-047 | True | True | True | True | 100% | 6.65s |
| recipe-048 | True | True | True | True | 100% | 6.83s |
| recipe-049 | True | True | True | True | 100% | 4.75s |
| recipe-050 | True | True | True | True | 100% | 4.97s |
| recipe-051 | True | True | True | True | 100% | 4.53s |
| recipe-052 | True | True | True | True | 100% | 5.92s |
| recipe-053 | True | True | True | True | 100% | 4.12s |
| recipe-054 | True | True | True | True | 100% | 6.06s |
