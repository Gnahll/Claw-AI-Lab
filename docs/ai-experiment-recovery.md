# AI 自主复试与断点恢复

`backend/services/ai_experiment_loop.py` 监控登记过的外部实验。已接入 OSGK 深度序关系（`osgk_depth_order`）、CIRB 条件增量（`cirb_mask_boundary`）和 GSF 谱融合（`gsf_spectral`）。每个项目保留各自的训练、基线和结果。

在项目根目录添加 `ai_experiment_policy.json`：

```json
{
  "enabled": true,
  "family": "cirb_mask_boundary",
  "revision_mode": "ai_authored_formulas",
  "initial_run": "run-1350步短训",
  "initial_unit": "claw-cirb-fixed-score-mask-1126",
  "scenes": 312,
  "min_ap_gain": 0.5,
  "max_ap25_drop": 0.1,
  "max_retries_per_run": 3,
  "api_retry_seconds": 600
}
```

```bash
python -B backend/services/ai_experiment_loop.py --repo "$PWD" --projects "$PWD/backend/runs/projects"
```

模型沿用平台 `.env` 中的 API。每次短训未达标后，它获得实际指标、近期梯度与边缘修正统计、已试过的改动以及上次公式校验反馈。模型自己提出原因、可证伪假设和一至两条张量表达式，CIRB 可改邻域条件、消息聚合、掩码残差或训练损失。OSGK 可改邻域亲和力、深度序偏置、近远上下文和残差增益。GSF 可改谱差值的表达和残差增益。模型可基于原方法或在现有公式上继续调整。

平台校验表达式的语法、允许的变量和算子、张量形状、有限性、需要时的梯度和零初始化，随后调用原训练入口的真实预检。重复表达式和无效公式写入状态，其中无效公式会交回模型重新编写。项目里的训练预算、验证集和验收门槛由登记配置控制。

OSGK 复用同预算的四尺度评估协议。GSF 同时比较配对对照和冻结锚点。每轮另建运行目录。`ai_revision.json` 保存公式和实际代码哈希，`ai_experiment_history.json` 保存历轮原因、假设、指标和进度。普通中断从已有检查点恢复，GPU 忙时等待。磁盘不足、显存不足、代码错误和坏检查点记录后暂停。通过验收时沿用实验自带的长训接续。可用 `max_revisions` 限制复试次数，默认随实验继续。将 `enabled` 设为 `false` 停止产生新任务，已启动的训练继续运行。
