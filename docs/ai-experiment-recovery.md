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

## 机制预检与短训预算

新方案必须给出要区分的原因、对应的公式观测点、预期变化、短训档位和选择理由。平台使用训练批次的前 8 步比较新旧公式在同一输入上的信号，并测量各方向的对照是否可区分。通过的前 8 步计入短训。无有效信号或等价对照的方案会停止在预检结果，AI 读取 `preflight_mechanism.json` 后重新提出改法。

| 方向 | 快筛 | 对照 | 确认 |
| --- | ---: | ---: | ---: |
| OSGK 从已有节点继续 | +150 | +300 | +450 |
| CIRB、GSF 每组新增 | 450 | 900 | 1,350 |

每轮方法组与对照使用同一预算，所有拟接长训的结果仍按 312 个场景和原验收指标判断。

## 长训前的短训审核

普通流水线在 S9 的 `exp_plan.yaml` 声明 `training_phase: long` 和训练预算。S14 执行和 S15 重跑使用同一监督器。已知长训先在隔离目录运行同源 PyTorch 短训；未声明预算的受监督 PyTorch 任务也有按进程实际优化步数的上限。工程冒烟测试与方法验证分别记录。生成的训练入口通过 `RESEARCHCLAW_PILOT_UPDATES` 接收每个 rank 的更新预算，并按冻结的计划完成验证。已有外部训练脚本若尚未接入该预算与产物协议，平台会报告缺少的证据。

S9 可在 `short_training_review` 中设置 `pilot_steps`、`validation_samples` 与 `comparisons`。每项比较填写 `method`、`control`、`metric`、`direction` 和 `min_gain`。原生模式默认短训 150 步，动态预算的保护上限为 1350 步。每个项目使用自己的指标与单位。审批记录关联实际参数更新、检查点、预测、验证样本数和对照结果。有稳定输入指纹的同一运行可以复用实测产物，重新请求 AI 审核。

独立 Co-Transformers 任务可在训练规格中声明 `training_phase: long`，可选 `pilot_steps` 和 `min_validation_f1_gain`（F1 取值 0～1）。计划总更新超过 1350 步同样触发审核。它以原初始化在训练集完成有预算的试验，对同一验证集记录训练前后 pixel F1 和 IoU；结果见任务目录 `long_review/`。通过后按原协议启动或续跑主任务，验证用的测试不消耗测试集。

新生成的 OSGK、CIRB、GSF 调度脚本在短训与长训的交接点依次验证门槛、同预算四卡账本、配对样本与帧、检查点哈希、内置 AI 意见。`long_review_request.json` 和 `long_review_decision.json` 保留证据、绑定指纹及决定。项目策略里规定的最低收益仍然生效。

## 输入文件格式恢复

平台在受支持的阶段输入和已登记实验的运行目录内尝试恢复指定 JSON 数据（单个 UTF-8 BOM、完整 JSON 代码围栏、唯一尾逗号），以及 `search_plan.yaml`、`exp_plan.yaml` 的完整 YAML 围栏。内置 AI 确认经解析器验证的单项修改后，平台备份原始字节，核对文件未被并发修改，原子替换并重试原读取入口。已冻结计划还需匹配原摘要。`.artifact-format-recovery/` 记录原文、哈希、动作与重试。服务暂时不可用时按退避时间重试，含糊输入、检查点、权重和研究指标交给项目负责人处理。
