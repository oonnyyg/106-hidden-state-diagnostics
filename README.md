# Hidden State Diagnostics

设备巡检用的离散隐马尔可夫（HMM）诊断工具。给定**固定**的状态模型与多条带标识的观测序列，
输出每条序列最可能的完整隐藏状态路径（Viterbi）与整条观测的对数似然（前向算法）。
不进行任何参数训练，仅依赖 Python 标准库（Python ≥ 3.12）。

## 模型 JSON 格式

```json
{
  "states": ["normal", "worn", "fault"],
  "observations": ["vibration", "heat", "noise"],
  "initial":     {"normal": 0.8, "worn": 0.15, "fault": 0.05},
  "transitions": {
    "normal": {"normal": 0.85, "worn": 0.12, "fault": 0.03},
    "worn":   {"normal": 0.10, "worn": 0.70, "fault": 0.20},
    "fault":  {"normal": 0.05, "worn": 0.15, "fault": 0.80}
  },
  "emissions": {
    "normal": {"vibration": 0.70, "heat": 0.20, "noise": 0.10},
    "worn":   {"vibration": 0.20, "heat": 0.50, "noise": 0.30},
    "fault":  {"vibration": 0.10, "heat": 0.20, "noise": 0.70}
  },
  "sequences": [
    {"id": "line-A", "observations": ["vibration", null, "heat"]}
  ]
}
```

校验规则：

- `states` 与 `observations` 均为非空、唯一的字符串。
- `initial`、每个 `transitions[state]`、每个 `emissions[state]` 都必须覆盖且仅覆盖
  对应的全部键；概率必须是有限数值且在 `[0, 1]` 闭区间内。
- 每个概率分布之和与 1 的绝对误差不得超过 `1e-12`（万亿分之一）。
- **零概率表示不可能事件**。首时刻直接使用 `initial` 分布，不做转移。
- 观测为 `null` 或空字符串 `""` 表示该时刻**缺测**：不施加发射约束（发射概率视为 1），
  但状态转移照常发生。
- 空观测序列（`[]`）被明确拒绝，状态为 `empty_sequence`。

## 数值处理

所有概率在加载后立即转换为对数概率，零概率映射为 `-inf`：

- **前向算法**在对数空间递推，多路径求和使用 log-sum-exp 稳定实现，得到
  `log P(O|λ)`；长序列不会因概率连乘下溢而被误判为不可能。
- **Viterbi** 在对数空间做全局动态规划，最大化整条路径的联合概率
  `P(path, O|λ)`，不是逐步贪心。并列最优路径按完整状态名称序列的**字典序**取最小者
  （按字典序遍历状态，仅在严格更优时更新）。
- 当一条序列的总概率为 0（所有路径均被零初始/零转移/零发射封死）时，
  返回状态 `impossible` 和独立失败原因，`log_likelihood` 与 `path` 均为 `null`，
  不编造路径，也不影响同批其他序列的分析。

## 输出结构

每条序列的结果保持输入顺序，包含：

- `status`：`ok` / `impossible` / `empty_sequence` / `invalid_sequence`
- `log_likelihood`：前向算法得到的整条观测对数似然
- `path`：Viterbi 完整状态路径（逐时刻状态）
- `path_log_probability`：该路径与观测的联合对数概率
- `state_counts`：路径中各状态的占用次数
- `transition_counts`：路径中 `源状态 -> 目标状态` 的转移次数矩阵
- `reason`：失败时的原因说明

顶层 `summary` 给出总数、各状态计数，以及所有成功序列的 `log_likelihoods`。

## 命令行用法

```bash
python -m hmm_diagnostics examples/inspection_model.json
python -m hmm_diagnostics model.json -o report.json
```

退出码：全部成功为 `0`；存在未成功分析的序列为 `1`；模型文件无法加载或校验失败为 `2`。

## 作为库使用

```python
from hmm_diagnostics.model import load_model
from hmm_diagnostics.diagnose import analyze_all, build_report

model = load_model("model.json")
report = build_report(model, analyze_all(model))
```

## 测试

```bash
python -m pytest tests/
```

覆盖：缺测（null/空串）仍发生转移、零初始/零发射导致 `impossible`、
逐步贪心会选错而 Viterbi 选全局最优的反例、完整路径字典序并列、
与暴力枚举一致性、5000 步长序列防下溢、空序列拒绝与输入顺序保持。
