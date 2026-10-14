# train — 能力训练

`Trainer(backend, lr)` 把 `skill` 单调爬向 1（`skill += lr * (1 - skill)`），并记录
`(epoch, skill, hit_rate)` 历史。命中率随 skill 单调上升，是 LES 价值的直接度量。

**铁律**：信号单调趋 1，不要「对 +1 错 -1」，否则低能力时期望漂移为负。
