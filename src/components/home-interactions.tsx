'use client';

import Link from 'next/link';
import { ArrowRight, BookOpen, Braces, Code2 } from 'lucide-react';
import { useState } from 'react';

const stages = [
  {
    key: 'rollout',
    label: '① 学生采样',
    note: '学生把「因此」作为下一 token 的最高概率选择。',
    student: [62, 21, 11],
    teacher: [34, 43, 18],
  },
  {
    key: 'feedback',
    label: '② 教师评分',
    note: '老师更偏好「所以」；log-ratio 给出移动方向。',
    student: [62, 21, 11],
    teacher: [34, 43, 18],
  },
  {
    key: 'update',
    label: '③ 更新之后',
    note: '学生仍保留自己的分布，但向老师的局部判断靠近。',
    student: [49, 33, 13],
    teacher: [34, 43, 18],
  },
];

const tokens = ['因此', '所以', '接着'];

export function HeroConsole() {
  const [active, setActive] = useState(0);
  const stage = stages[active];

  return (
    <div className="hero-console">
      <div className="console-chrome">
        <div><i /><i /><i /></div>
        <span>student_state / token_047</span>
        <b>LIVE</b>
      </div>
      <div className="console-prompt">
        <span>PROMPT</span>
        <p>如果训练数据来自老师，而部署时文本来自学生自己，会发生什么？</p>
      </div>
      <div className="distribution-head">
        <span>TOKEN</span><span>STUDENT πθ</span><span>TEACHER πT</span>
      </div>
      <div className="distribution-list">
        {tokens.map((token, index) => (
          <div className="distribution-row" key={token}>
            <strong>{token}</strong>
            <div className="bar-cell"><i className="student-bar" style={{ width: `${stage.student[index]}%` }} /><em>{stage.student[index]}%</em></div>
            <div className="bar-cell"><i className="teacher-bar" style={{ width: `${stage.teacher[index]}%` }} /><em>{stage.teacher[index]}%</em></div>
          </div>
        ))}
      </div>
      <div className="console-note"><span>Δ</span><p>{stage.note}</p></div>
      <div className="console-steps" role="tablist" aria-label="OPD 三步演示">
        {stages.map((item, index) => (
          <button
            key={item.key}
            role="tab"
            aria-selected={index === active}
            onClick={() => setActive(index)}
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}

const paths = [
  {
    key: 'beginner',
    label: '第一次接触',
    icon: BookOpen,
    title: '90 分钟建立完整心智模型',
    steps: ['三分钟认识 OPD', 'SFT / KD / RL 对照', 'KL 方向交互实验', '最小训练循环'],
    href: '/docs/start/mental-model',
  },
  {
    key: 'math',
    label: '想吃透算法',
    icon: Braces,
    title: '从序列 KL 推到实际估计器',
    steps: ['状态分布与 exposure bias', 'Forward / Reverse KL', 'sampled / top-k / full', '失败机制与新方法'],
    href: '/docs/foundations/kl-divergence',
  },
  {
    key: 'engineer',
    label: '准备跑代码',
    icon: Code2,
    title: '沿两套实现完成第一次实验',
    steps: ['tinker 最小 recipe', 'verl 角色与数据协议', '配置与显存预算', '监控、排错、复现记录'],
    href: '/docs/engineering/tinker-cookbook',
  },
];

export function PathSelector() {
  const [active, setActive] = useState('beginner');
  const path = paths.find((item) => item.key === active) ?? paths[0];

  return (
    <div className="path-picker">
      <div className="path-tabs" role="tablist" aria-label="选择学习路线">
        {paths.map(({ key, label, icon: Icon }) => (
          <button key={key} role="tab" aria-selected={key === active} onClick={() => setActive(key)}>
            <Icon size={16} /> {label}
          </button>
        ))}
      </div>
      <div className="path-panel">
        <span>RECOMMENDED ROUTE</span>
        <h3>{path.title}</h3>
        <ol>
          {path.steps.map((step, index) => <li key={step}><b>{index + 1}</b>{step}</li>)}
        </ol>
        <Link href={path.href}>从这里出发 <ArrowRight size={16} /></Link>
      </div>
    </div>
  );
}
