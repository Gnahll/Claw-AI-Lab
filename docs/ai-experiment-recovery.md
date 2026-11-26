# AI 实验诊断与恢复

`backend/services/ai_experiment_loop.py` 处理显式登记的外部实验。首次启用支持 CIRB 固定评分掩码边缘实验。平台原来的阶段队列可继续使用。

在项目根目录设置 `ai_experiment_policy.json`：

```json
{
  "enabled": true,
  "family": "cirb_mask_boundary",
  "initial_run": "run-1350步短训",
  "initial_unit": "claw-cirb-fixed-score-mask-1126",
  "scenes": 312,
  "min_ap_gain": 0.5,
  "max_ap25_drop": 0.1,
  "max_retries_per_run": 3,
  "api_retry_seconds": 600,
  "variants": ["protect_good_masks", "reduce_boundary_displacement", "emphasize_iou"]
}
```

用平台的 Python 环境启动服务：

```bash
python -B backend/services/ai_experiment_loop.py --repo "$PWD" --projects "$PWD/backend/runs/projects"
```

模型沿用平台 `.env` 的 API。模型获得评估分数、故障类别、训练进度和先前尝试，返回诊断与一个登记过的动作。模型没有命令执行入口。API 暂时不可用时，状态记为 `await_ai`，由服务稍后重试。

项目内的 `ai_experiment_history.json` 记录每次验收、故障、AI 诊断、假设和执行动作。每个新方法单独建立一个运行目录，`ai_revision.json` 记录实际代码改动、预算和基线。脚本比较统一场景上的主 AP 与 AP25。通过后沿用实验自带的长训接续程序。每种登记方法最多尝试一次。GPU 争用会等待资源，普通中断通过原训练程序读取检查点。磁盘故障、显存不足、代码错误和损坏权重需要明确处理，状态与诊断会保留在记录中。

将 `enabled` 改为 `false` 可以禁止后续自动动作。已经启动的训练继续运行。更换实验类型时需登记相应的受控方法和恢复入口。
