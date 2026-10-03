# Hidden State Diagnostics

设备巡检离散隐马尔可夫（HMM）诊断工具。给定已训练好的状态模型，对多条带标识的
观测序列计算对数似然（前向算法）并解码最可能的完整隐状态路径（Viterbi）。
仅使用 Python 标准库，不做模型训练。

## 使用方法

```bash
python -m hmm_diag input.json [-o output.json]
```

不提供 `-o` 时结果输出到 stdout。模型非法或输入文件无法读取时退出码为 2；
单条序列的失败不会中断同批其他序列。

运行测试：

```bash
python -m unittest discover -s tests
```

## 输入格式

```json
{
  "model": {
    "states": ["Healthy", "Fault"],
    "observations": ["normal", "alarm"],
    "initial": {"Healthy": 0.8, "Fault": 0.2},
    "transition": {
      "Healthy": {"Healthy": 0.9, "Fault": 0.1},
      "Fault": {"Healthy": 0.3, "Fault": 0.7}
    },
    "emission": {
      "Healthy": {"normal": 0.9, "alarm": 0.1},
      "Fault": {"normal": 0.2, "alarm": 0.8}
    }
  },
  "sequences": [
    {"id": "run-1", "observations": ["normal", null, "alarm"]}
  ]
}
```

- `states` / `observations`：名称必须为非空字符串且各自唯一。
- `initial`：初始状态分布，首时刻直接使用。
- `transition`：每行是一个完整的下一状态分布。
- `emission`：每个状态对全部观测符号的发射分布。
- 概率必须是有限数值且落在 `[0, 1]`；每个分布之和与 1 的误差不得超过
  `1e-12`。`0` 表示不可能事件，是合法取值。
- 观测序列中 `null`（或空字符串 `""`）表示该时刻缺测：没有发射约束，
  但状态转移照常发生。空序列（`[]`）会被明确拒绝。
- 未知观测符号、缺失/非法的 `id` 会使该条序列标记为 `failed`，并给出原因。

完整示例见 `examples/inspection.json`。

## 输出格式

结果按输入顺序逐条返回：

- `status`：`ok` / `impossible`（模型下零概率，不编造路径）/ `failed`（输入非法）。
- `log_likelihood`：整条观测序列的对数似然（前向算法）。
- `path` / `path_log_probability`：联合概率最大的完整状态路径及其对数概率。
- `timeline`：逐时刻的观测与解码状态。
- `state_counts`：各状态占用次数；`transition_counts`：各状态间转移次数。
- `summary`：总数及 `ok` / `impossible` / `failed` 计数。

## 数值处理

- 所有计算在对数空间进行：前向算法用 log-sum-exp 累加，Viterbi 用对数概率
  取最大，长序列不会因概率连乘下溢而被误判为不可能。
- 零概率映射为 `-inf`，不可能的路径被自然排除；整条序列零概率时返回
  `impossible` 与失败原因。
- Viterbi 求的是完整路径的联合最优，而非每步局部最优；并列最优时按完整
  状态名称序列的字典序选择。
