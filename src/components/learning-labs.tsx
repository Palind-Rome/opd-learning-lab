'use client';

import { useMemo, useState } from 'react';
import { ArrowRight, Check, CircleAlert, RotateCcw, Sparkles } from 'lucide-react';

function kl(p: number, q: number) {
  const eps = 1e-8;
  return p * Math.log((p + eps) / (q + eps)) + (1 - p) * Math.log((1 - p + eps) / (1 - q + eps));
}

export function KLDivergenceLab() {
  const [student, setStudent] = useState(0.72);
  const [teacher, setTeacher] = useState(0.36);
  const reverse = kl(student, teacher);
  const forward = kl(teacher, student);
  const midpoint = (student + teacher) / 2;
  const js = (kl(student, midpoint) + kl(teacher, midpoint)) / 2;

  return (
    <div className="learning-lab kl-lab">
      <div className="lab-heading">
        <div><span>INTERACTIVE 01</span><h3>亲手拨动两条分布</h3></div>
        <button onClick={() => { setStudent(.72); setTeacher(.36); }}><RotateCcw size={14} /> 重置</button>
      </div>
      <div className="kl-controls">
        <label>
          <span><i className="dot student-dot" /> 学生把多少概率给 token A？ <b>{Math.round(student * 100)}%</b></span>
          <input type="range" min="5" max="95" value={Math.round(student * 100)} onChange={(event) => setStudent(Number(event.target.value) / 100)} />
        </label>
        <label>
          <span><i className="dot teacher-dot" /> 教师把多少概率给 token A？ <b>{Math.round(teacher * 100)}%</b></span>
          <input type="range" min="5" max="95" value={Math.round(teacher * 100)} onChange={(event) => setTeacher(Number(event.target.value) / 100)} />
        </label>
      </div>
      <div className="binary-distributions">
        <div><span>学生 πθ</span><i style={{ width: `${student * 100}%` }} /><b>A {Math.round(student * 100)}%</b><em>B {Math.round((1 - student) * 100)}%</em></div>
        <div><span>教师 πT</span><i style={{ width: `${teacher * 100}%` }} /><b>A {Math.round(teacher * 100)}%</b><em>B {Math.round((1 - teacher) * 100)}%</em></div>
      </div>
      <div className="metric-grid">
        <article><span>Reverse KL</span><strong>{reverse.toFixed(3)}</strong><code>D(πθ ‖ πT)</code></article>
        <article><span>Forward KL</span><strong>{forward.toFixed(3)}</strong><code>D(πT ‖ πθ)</code></article>
        <article><span>Jensen–Shannon</span><strong>{js.toFixed(3)}</strong><code>½ KL(πθ‖m)+½ KL(πT‖m)</code></article>
      </div>
      <p className="lab-footnote"><CircleAlert size={14} /> 二元玩具分布只展示“方向不对称”；mode-seeking / mode-covering 要在多峰分布中理解，不能只凭这两个数字下结论。</p>
    </div>
  );
}

const granularities = {
  sampled: {
    label: 'Sampled token',
    visible: 1,
    payload: '每个位置只取学生实际采到的 token',
    cost: '教师返回 1 个目标 token 的 log-prob',
    estimator: '在固定前缀、精确采样条件下，可作 per-token RKL 的单样本估计；方差较高',
    caveat: '别把“无偏”扩展成整个异步训练系统都无偏。',
  },
  topk: {
    label: 'Top-k',
    visible: 5,
    payload: '保留局部候选集，再截断或重新归一化',
    cost: '教师返回 k 个 token 的分数与索引',
    estimator: '在成本和多 token 信号之间折中；具体偏差取决于支持集和归一化方式',
    caveat: 'teacher top-k、student top-k 与二者并集不是同一个目标。',
  },
  full: {
    label: 'Full vocabulary',
    visible: 12,
    payload: '每个位置比较完整词表分布',
    cost: '激活/通信规模随 B × T × |V| 增长',
    estimator: '对给定学生前缀可直接计算完整 token-level divergence，信号最密',
    caveat: '“最完整”不等于端到端一定最好；显存、带宽和 teacher 吞吐会先成为瓶颈。',
  },
} as const;

type Granularity = keyof typeof granularities;

export function GranularityLab() {
  const [mode, setMode] = useState<Granularity>('sampled');
  const item = granularities[mode];

  return (
    <div className="learning-lab granularity-lab">
      <div className="lab-heading"><div><span>INTERACTIVE 02</span><h3>教师到底要返回多少信息？</h3></div></div>
      <div className="segmented" role="tablist" aria-label="监督粒度">
        {(Object.keys(granularities) as Granularity[]).map((key) => (
          <button key={key} role="tab" aria-selected={mode === key} onClick={() => setMode(key)}>{granularities[key].label}</button>
        ))}
      </div>
      <div className="vocab-strip" aria-label={`${item.label} 可见 token 示意`}>
        {Array.from({ length: 12 }, (_, index) => <i key={index} data-visible={index < item.visible || undefined}>{index === 0 && mode === 'sampled' ? 'ŷ' : index + 1}</i>)}
      </div>
      <div className="granularity-detail">
        <article><span>看见什么</span><p>{item.payload}</p></article>
        <article><span>付出什么</span><p>{item.cost}</p></article>
        <article><span>得到什么</span><p>{item.estimator}</p></article>
      </div>
      <p className="lab-footnote"><CircleAlert size={14} /> {item.caveat}</p>
    </div>
  );
}

const cycleSteps = [
  { title: 'Prompt batch', owner: 'Data', detail: '只取 prompt；不要把数据集里的标准 response 当本轮 rollout。', artifact: 'input_ids / attention_mask' },
  { title: 'Student rollout', owner: 'Rollout', detail: '当前学生按解码配置生成 response，形成真正 on-policy 的前缀。', artifact: 'response_ids / response_mask' },
  { title: 'Teacher scoring', owner: 'Teacher', detail: '教师在完全相同的 prompt + 学生 response 上做 teacher forcing。', artifact: 'teacher_log_probs' },
  { title: 'Student scoring', owner: 'Actor', detail: '学生重新计算同一批 token 的 log-prob；注意 shift 与 mask 对齐。', artifact: 'old_log_probs / log_probs' },
  { title: 'Loss & update', owner: 'Actor', detail: '构造 KL 或 log-ratio 信号，聚合有效 token，反向传播并刷新 rollout 权重。', artifact: 'loss / metrics / checkpoint' },
];

export function OPDCycle() {
  const [active, setActive] = useState(0);
  const step = cycleSteps[active];
  return (
    <div className="learning-lab cycle-lab">
      <div className="lab-heading"><div><span>INTERACTIVE 03</span><h3>一次训练 step 的五个交接点</h3></div></div>
      <div className="cycle-track" role="tablist" aria-label="OPD 训练循环">
        {cycleSteps.map((item, index) => (
          <button key={item.title} role="tab" aria-selected={index === active} onClick={() => setActive(index)}>
            <b>{index + 1}</b><span>{item.title}</span>{index < cycleSteps.length - 1 && <ArrowRight size={14} />}
          </button>
        ))}
      </div>
      <div className="cycle-detail">
        <span>{step.owner} OWNS THIS STEP</span>
        <h4>{step.title}</h4>
        <p>{step.detail}</p>
        <code>{step.artifact}</code>
      </div>
    </div>
  );
}

export function CompatibilityLab() {
  const [overlap, setOverlap] = useState(55);
  const [novelty, setNovelty] = useState(55);
  const [depth, setDepth] = useState(50);
  const result = useMemo(() => {
    if (overlap < 35) return { tone: 'risk', title: '先别直接开跑', text: '局部候选重合很低。优先检查 tokenizer / chat template / special token，并考虑 cold start 或更兼容的 teacher。' };
    if (novelty < 30) return { tone: 'watch', title: '可能没有新东西可学', text: '老师更高分不等于分布里含有学生未见的能力。先做小规模 probe，再决定是否投入完整训练。' };
    if (depth > 75) return { tone: 'watch', title: '长轨迹需要额外监控', text: '后段 prefix drift 可能让局部监督变差。记录分位置 KL / overlap，并考虑截断或位置权重。' };
    return { tone: 'ready', title: '适合进入小规模试跑', text: '这只是工程启发式，不是成功判定公式。先跑短实验，检查 overlap、entropy、长度和下游指标是否一起健康。' };
  }, [overlap, novelty, depth]);

  return (
    <div className="learning-lab compatibility-lab">
      <div className="lab-heading"><div><span>INTERACTIVE 04</span><h3>开跑前的三轴体检</h3></div></div>
      <div className="compat-controls">
        <label><span>局部候选重合（proxy）<b>{overlap}</b></span><input type="range" min="0" max="100" value={overlap} onChange={(e) => setOverlap(Number(e.target.value))} /></label>
        <label><span>教师新增能力（probe）<b>{novelty}</b></span><input type="range" min="0" max="100" value={novelty} onChange={(e) => setNovelty(Number(e.target.value))} /></label>
        <label><span>目标轨迹长度压力<b>{depth}</b></span><input type="range" min="0" max="100" value={depth} onChange={(e) => setDepth(Number(e.target.value))} /></label>
      </div>
      <div className="compat-result" data-tone={result.tone}><Sparkles size={18} /><div><h4>{result.title}</h4><p>{result.text}</p></div></div>
      <p className="lab-footnote"><CircleAlert size={14} /> 三个滑块是把论文诊断变成检查清单，不是作者提出的统一分数，也没有通用阈值。</p>
    </div>
  );
}

const quiz = [
  {
    question: '哪一项最准确地描述 “on-policy”？',
    options: ['必须使用 PPO clip', '回答/前缀来自当前学生策略', '教师和学生必须同架构'],
    answer: 1,
    explain: 'On-policy 描述的是数据状态分布来自谁；它和用哪一种 KL、是否使用 PPO clip 是不同的设计轴。',
  },
  {
    question: '教师总体 benchmark 更强，是否保证 OPD 有效？',
    options: ['保证', '不保证', '只要 teacher 更大就保证'],
    answer: 1,
    explain: 'Rethinking OPD 等工作显示，局部 thinking-pattern compatibility 与真正的新能力同样关键。',
  },
  {
    question: 'sampled-token OPD 中最容易错的一处工程细节是？',
    options: ['网页颜色', 'response shift 与 mask 对齐', '把 batch size 写成偶数'],
    answer: 1,
    explain: 'teacher/student 必须给同一目标 token 打分；一位错位或把 prompt/padding 算进 loss 都会改变目标。',
  },
];

export function KnowledgeCheck() {
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const answered = Object.keys(answers).length;
  const correct = quiz.reduce((sum, item, index) => sum + (answers[index] === item.answer ? 1 : 0), 0);
  return (
    <div className="learning-lab quiz-lab">
      <div className="lab-heading"><div><span>CHECKPOINT</span><h3>三题确认你真的理解了</h3></div><b>{answered === quiz.length ? `${correct} / ${quiz.length}` : `${answered} / ${quiz.length}`}</b></div>
      <div className="quiz-list">
        {quiz.map((item, index) => (
          <article key={item.question}>
            <h4><span>{index + 1}</span>{item.question}</h4>
            <div>{item.options.map((option, optionIndex) => (
              <button key={option} data-picked={answers[index] === optionIndex || undefined} data-correct={answers[index] !== undefined && optionIndex === item.answer || undefined} onClick={() => setAnswers((old) => ({ ...old, [index]: optionIndex }))}>{option}</button>
            ))}</div>
            {answers[index] !== undefined && <p><Check size={14} /> {item.explain}</p>}
          </article>
        ))}
      </div>
    </div>
  );
}
