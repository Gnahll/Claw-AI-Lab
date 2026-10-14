import { useEffect, useRef, useState } from 'react';
import type { ManagedTrainingSpec, ManagedTrainingView } from '../types';
import { useLocale } from '../i18n';

interface Props {
  connected: boolean;
  view: ManagedTrainingView;
  onCommand: (command: Record<string, unknown>) => void;
}

export default function TrainingPanel({ connected, view, onCommand }: Props) {
  const { locale } = useLocale();
  const chinese = locale === 'zh';
  const text = (zh: string, en: string) => chinese ? zh : en;
  const [open, setOpen] = useState(false);
  const [overrides, setOverrides] = useState<Partial<ManagedTrainingSpec>>({});
  const [topicOverride, setTopicOverride] = useState<string | null>(null);
  const [protocolApproved, setProtocolApproved] = useState(false);
  const [gpuApproved, setGpuApproved] = useState(false);
  const [gpuUuid, setGpuUuid] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [pending, setPending] = useState<{ id: string; expires: number } | null>(null);
  const [clock, setClock] = useState(0);
  const sendRef = useRef(onCommand);
  useEffect(() => { sendRef.current = onCommand; }, [onCommand]);

  useEffect(() => {
    if (!open || !connected) return;
    const poll = () => {
      setClock(Date.now());
      sendRef.current({ command: 'training_list' });
    };
    poll();
    const timer = setInterval(poll, 5000);
    return () => clearInterval(timer);
  }, [open, connected]);

  const spec = view.defaults ? { ...view.defaults, ...overrides } : undefined;
  const topic = topicOverride ?? view.default_topic ?? '';
  const busy = connected && pending !== null && pending.id !== view.operation_request_id && clock < pending.expires;
  const job = view.jobs.find((item) => item.id === (selectedId || view.last_job_id)) ?? view.jobs[0];
  const preflight = view.preflight;
  const reviewed = Boolean(preflight && spec && preflight.topic === topic.trim()
    && Object.entries(spec).every(([key, value]) => preflight.spec[key as keyof ManagedTrainingSpec] === value));
  const result = job?.verification?.status === 'passed' ? job.verification.result : undefined;

  const send = (command: string, values: Record<string, unknown> = {}) => {
    const identifier = crypto.randomUUID();
    setClock(Date.now());
    setPending({ id: identifier, expires: Date.now() + 180000 });
    onCommand({ command, requestId: identifier, ...values });
  };

  const edit = (key: keyof ManagedTrainingSpec, value: string | number) => {
    setOverrides((previous) => ({ ...previous, [key]: value }));
    setProtocolApproved(false);
  };

  const artifact = (filename: string) => `/download-training/${job?.id}/${filename}`;
  const statusText: Record<string, string> = {
    prepared: text('已准备，尚未训练', 'Prepared, not started'),
    starting: text('正在启动 GPU 训练', 'Starting GPU training'),
    running: text('正在训练／验证／测试', 'Training / validation / test running'),
    completed: text('训练与测试结束，仍需核验通过', 'Training/test complete; evidence verification required'),
    failed: text('训练失败', 'Training failed'),
    interrupted: text('训练中断', 'Training interrupted'),
    unreadable: text('任务文件异常', 'Job files unreadable'),
  };

  return (
    <section className="training-panel">
      <button className="training-heading" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} {text('真实训练验收（GPU）', 'Real Training Acceptance (GPU)')}
      </button>
      {open && <div className="training-body">
        <p>{text('独立验收入口：使用现有 Co-Transformers，最多 1 个截断训练轮次、3 次参数更新，并完成验证、测试和证据核验。不是完整论文复现，也不会自动运行论文写作。',
          'Dedicated acceptance: existing Co-Transformers, at most one truncated epoch / three optimizer steps, then validation, test and evidence verification. Not full paper reproduction or automatic paper writing.')}</p>
        {!connected && <p role="alert">{text('请连接服务器；演示模式不能执行真实训练。', 'Connect to the server; demo mode cannot run real training.')}</p>}
        <label>{text('研究课题（作为任务记录，不自动改模型）', 'Research topic (job record, not model modification)')}
          <textarea value={topic} maxLength={4000} onChange={(event) => {
            setTopicOverride(event.target.value); setProtocolApproved(false);
          }} />
        </label>
        {spec && <>
          {([
            ['source_dir', '代码目录', 'Code directory'],
            ['dataset_root', '数据目录', 'Dataset directory'],
            ['split_manifest', '已审计的数据划分文件', 'Audited split manifest'],
            ['initial_checkpoint', '初始权重文件', 'Initial checkpoint'],
            ['initial_checkpoint_sha256', '初始权重 SHA-256', 'Initial checkpoint SHA-256'],
          ] as const).map(([key, zh, en]) => <label key={key}>{text(zh, en)}
            <input value={spec[key]} onChange={(event) => edit(key, event.target.value)} />
          </label>)}
          <label>{text('参数更新次数（1–3）', 'Optimizer steps (1–3)')}
            <input type="number" min={1} max={3} value={spec.steps_per_epoch}
              onChange={(event) => edit('steps_per_epoch', Number(event.target.value))} />
          </label>
        </>}
        <button disabled={!connected || busy || !spec || !topic.trim()}
          onClick={() => send('training_preflight', { spec, topic })}>
          {text('1. 检查输入（不启动训练）', '1. Check inputs (no training)')}
        </button>
        {preflight && reviewed && <div className="training-review">
          <p>{text('检查通过：训练／验证／测试', 'Checked: train / validation / test')} =
            {preflight.counts.train} / {preflight.counts.validation} / {preflight.counts.test}</p>
          <p>{text('此前已评估的组', 'Previously evaluated groups')}: {preflight.previously_evaluated_groups.join(', ') || '—'}</p>
          <label className="training-confirm"><input type="checkbox" checked={protocolApproved}
            onChange={(event) => setProtocolApproved(event.target.checked)} />
            {text('我批准这份划分用于 COVERAGE 小规模微调；它不是原论文完整复现，也不是未接触的盲测。',
              'I approve this split for COVERAGE pilot fine-tuning, not full paper reproduction or an untouched blind test.')}
          </label>
          <button disabled={!connected || busy || !protocolApproved}
            onClick={() => { setSelectedId(''); send('training_prepare', { preflightId: preflight.id, protocolApproved }); }}>
            {text('2. 准备真实训练任务（不占 GPU）', '2. Prepare real training job (no GPU use)')}
          </button>
        </div>}
        {view.jobs.length > 0 && <label>{text('训练任务', 'Training job')}
          <select value={job?.id ?? ''} onChange={(event) => {
            setSelectedId(event.target.value); setGpuApproved(false); setGpuUuid('');
          }}>
            {view.jobs.map((item) => <option key={item.id} value={item.id}>{item.topic} ({item.id.slice(0, 8)})</option>)}
          </select>
        </label>}
        {job && <div className="training-job">
          <p>{text('状态', 'Status')}: {statusText[job.state.status] ?? job.state.status}</p>
          <p>{text('阶段／更新步数', 'Phase / optimizer steps')}: {job.state.phase ?? '—'} / {job.state.global_step ?? 0} / {job.expected_steps ?? '—'}</p>
          <p>{text('GPU', 'GPU')}: {job.state.gpu_uuid ?? text('未启动', 'Not started')}</p>
          <code>{job.job_dir}</code>
          <button disabled={!connected || busy} onClick={() => send('training_gpus')}>
            {text('查看 GPU（仅查询，不分配）', 'Inspect GPUs (read-only, no allocation)')}
          </button>
          {view.gpus?.map((gpu) => <p key={gpu.uuid} className="training-gpu">{gpu.name}: {gpu.uuid}; {gpu.memory_mib} MiB; {gpu.utilization}%</p>)}
          <label>{text('管理员／调度器分配给你的 GPU UUID', 'GPU UUID allocated by administrator / scheduler')}
            <input value={gpuUuid} placeholder="GPU-..." onChange={(event) => { setGpuUuid(event.target.value); setGpuApproved(false); }} />
          </label>
          <label className="training-confirm"><input type="checkbox" checked={gpuApproved}
            onChange={(event) => setGpuApproved(event.target.checked)} />
            {text('我确认该 GPU 已获准由本任务独占使用；看起来空闲不等于已分配。',
              'I confirm exclusive allocation of this GPU to this job; appearing idle is not allocation.')}
          </label>
          <button disabled={!connected || busy || !gpuApproved || !gpuUuid.trim() || job.state.status !== 'prepared'}
            onClick={() => send('training_start', { jobId: job.id, gpuUuid: gpuUuid.trim(), gpuAllocated: gpuApproved })}>
            {text('3. 启动 GPU 真实训练', '3. Start real GPU training')}
          </button>
          {job.can_resume && <button disabled={!connected || busy || !gpuApproved || !gpuUuid.trim()}
            onClick={() => send('training_resume', { jobId: job.id, gpuUuid: gpuUuid.trim(), gpuAllocated: gpuApproved })}>
            {text('从已提交权重恢复（须用原 GPU）', 'Resume committed checkpoint (original GPU required)')}
          </button>}
          <button disabled={!connected || busy || job.state.status !== 'completed'}
            onClick={() => send('training_verify', { jobId: job.id })}>
            {text('重新核验真实训练证据', 'Re-verify real training evidence')}
          </button>
          {(job.state.error || job.state.alert || job.verification?.error) && <p role="alert">
            {job.state.error || job.state.alert || job.verification?.error}</p>}
          {result && <div className="training-passed">
            <strong>{text('真实训练核验通过', 'Real training verification passed')}</strong>
            <p>{text('上次核验', 'Last verification')}: {new Date((job.verification?.verified_at ?? 0) * 1000).toLocaleString()}</p>
            <p>F1: {result.metrics.pixel_f1.toFixed(6)} · IoU: {result.metrics.pixel_iou.toFixed(6)}</p>
            <p>{text('实际参数更新／测试图片', 'Actual optimizer steps / test images')}: {result.training_evidence.optimizer_steps} / {result.test_images}</p>
            <p>{text('已保存并重新加载的权重', 'Saved and reloaded checkpoint')}: {result.training_evidence.selected_checkpoint.filename}</p>
            <p>{text('参数哈希', 'Parameter hashes')}: {result.training_evidence.initial_parameter_sha256.slice(0, 12)} → {result.training_evidence.trained_parameter_sha256.slice(0, 12)}</p>
            <p>{text('S14 实验结果已导入，S15 测试集调参已禁用。', 'S14 evidence imported; S15 test-guided tuning disabled.')}</p>
            {['web_verification.json', 'test_results.json', 'events.jsonl', 'pipeline/stage-14/runs/run_report.json'].map((filename) =>
              <a key={filename} href={artifact(filename)} target="_blank" rel="noreferrer">{filename}</a>)}
            <details><summary>{text('全部测试掩码对比（左：真实；右：预测）', 'All test masks (left: target; right: prediction)')}</summary>
              {job.predictions?.map((record) => <div className="training-masks" key={record.id}>
                <span>{record.id}</span><img loading="lazy" alt={`target ${record.id}`} src={artifact(`test_predictions/${record.target}`)} />
                <img loading="lazy" alt={`prediction ${record.id}`} src={artifact(`test_predictions/${record.prediction}`)} />
              </div>)}
            </details>
          </div>}
          <details><summary>{text('训练日志（最新部分）', 'Training log (tail)')}</summary><pre>{job.log || '—'}</pre></details>
        </div>}
        {busy && <p role="status">{text('服务器正在处理，请稍候；检查大权重可能需要一些时间。', 'Server processing; checking large weights may take time.')}</p>}
        {view.error && <p role="alert">{view.error}</p>}
      </div>}
    </section>
  );
}
