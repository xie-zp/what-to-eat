# Agent 评测报告

- 检索模式：neo4j
- Planner：LLMRecipePlanner（失败时规则降级）
- 场景数量：54
- 任务成功率：100.0%
- 检索命中率：100.0%
- 相关性命中率：100.0%
- 约束满足率：100.0%
- 引用覆盖率：100.0%
- 平均延迟：5.94 秒
- Planner 调用次数：166
- LLM 直接规划比例：100.0%
- 规则降级比例：0.0%

| 场景 | 成功 | 检索命中 | 相关性 | 安全约束 | 引用覆盖 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| recipe-001 | True | True | True | True | 100% | 5.29s |
| recipe-002 | True | True | True | True | 100% | 4.38s |
| recipe-003 | True | True | True | True | 100% | 3.35s |
| recipe-004 | True | True | True | True | 100% | 4.41s |
| recipe-005 | True | True | True | True | 100% | 4.38s |
| recipe-006 | True | True | True | True | 100% | 4.83s |
| recipe-007 | True | True | True | True | 100% | 4.59s |
| recipe-008 | True | True | True | True | 100% | 3.29s |
| recipe-009 | True | True | True | True | 100% | 4.71s |
| recipe-010 | True | True | True | True | 100% | 4.99s |
| safe-001 | True | True | True | True | 100% | 3.45s |
| safe-002 | True | True | True | True | 100% | 9.28s |
| menu-001 | True | True | True | True | 100% | 16.94s |
| menu-002 | True | True | True | True | 100% | 56.45s |
| recipe-015 | True | True | True | True | 100% | 5.38s |
| recipe-016 | True | True | True | True | 100% | 4.43s |
| recipe-017 | True | True | True | True | 100% | 4.18s |
| recipe-018 | True | True | True | True | 100% | 4.48s |
| recipe-019 | True | True | True | True | 100% | 4.14s |
| recipe-020 | True | True | True | True | 100% | 5.09s |
| recipe-021 | True | True | True | True | 100% | 4.54s |
| recipe-022 | True | True | True | True | 100% | 4.18s |
| recipe-023 | True | True | True | True | 100% | 4.46s |
| recipe-024 | True | True | True | True | 100% | 4.24s |
| recipe-025 | True | True | True | True | 100% | 5.47s |
| recipe-026 | True | True | True | True | 100% | 4.67s |
| recipe-027 | True | True | True | True | 100% | 5.48s |
| recipe-028 | True | True | True | True | 100% | 4.38s |
| recipe-029 | True | True | True | True | 100% | 6.85s |
| recipe-030 | True | True | True | True | 100% | 4.13s |
| recipe-031 | True | True | True | True | 100% | 4.27s |
| recipe-032 | True | True | True | True | 100% | 4.68s |
| recipe-033 | True | True | True | True | 100% | 3.96s |
| recipe-034 | True | True | True | True | 100% | 4.40s |
| recipe-035 | True | True | True | True | 100% | 3.80s |
| recipe-036 | True | True | True | True | 100% | 4.54s |
| recipe-037 | True | True | True | True | 100% | 3.14s |
| recipe-038 | True | True | True | True | 100% | 3.96s |
| recipe-039 | True | True | True | True | 100% | 4.52s |
| recipe-040 | True | True | True | True | 100% | 4.06s |
| recipe-041 | True | True | True | True | 100% | 3.93s |
| recipe-042 | True | True | True | True | 100% | 5.99s |
| recipe-043 | True | True | True | True | 100% | 5.06s |
| recipe-044 | True | True | True | True | 100% | 4.68s |
| recipe-045 | True | True | True | True | 100% | 4.57s |
| recipe-046 | True | True | True | True | 100% | 5.68s |
| recipe-047 | True | True | True | True | 100% | 3.83s |
| recipe-048 | True | True | True | True | 100% | 4.21s |
| recipe-049 | True | True | True | True | 100% | 4.04s |
| recipe-050 | True | True | True | True | 100% | 4.11s |
| recipe-051 | True | True | True | True | 100% | 3.67s |
| recipe-052 | True | True | True | True | 100% | 5.18s |
| recipe-053 | True | True | True | True | 100% | 7.63s |
| recipe-054 | True | True | True | True | 100% | 10.30s |
