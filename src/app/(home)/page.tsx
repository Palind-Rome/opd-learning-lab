import Link from 'next/link';
import {
  ArrowRight,
  BookOpen,
  Braces,
  Check,
  CircleDot,
  FlaskConical,
  Network,
  Sparkles,
} from 'lucide-react';
import { HeroConsole, PathSelector } from '@/components/home-interactions';

const chapters = [
  {
    number: '01',
    title: '先建立直觉',
    text: '从 SFT、KD、RL 的差异开始，理解为什么“学生自己走过的状态”如此重要。',
    icon: BookOpen,
    href: '/docs/start/mental-model',
    tone: 'lime',
  },
  {
    number: '02',
    title: '把公式拆开',
    text: '把状态分布、KL 方向、采样估计器与 token mask 分层理解，不再被符号吓住。',
    icon: Braces,
    href: '/docs/core/objective',
    tone: 'violet',
  },
  {
    number: '03',
    title: '沿源码跑一遍',
    text: '从 tinker 的最小循环走到 verl 的分布式 teacher、rollout、actor 数据流。',
    icon: Network,
    href: '/docs/engineering/verl-architecture',
    tone: 'orange',
  },
  {
    number: '04',
    title: '做一次可信实验',
    text: '用兼容性诊断、监控指标和排错手册，知道结果为何有效，也知道何时别信。',
    icon: FlaskConical,
    href: '/docs/practice/first-run',
    tone: 'blue',
  },
];

const sources = [
  'Thinking Machines',
  'GKD · ICLR 2024',
  'tinker-cookbook',
  'verl source',
  'Rethinking OPD',
  'AwesomeOPD · 139 resources',
];

export default function HomePage() {
  return (
    <main className="lab-home">
      <section className="home-hero">
        <div className="hero-glow hero-glow-one" />
        <div className="hero-glow hero-glow-two" />
        <div className="home-shell hero-grid">
          <div className="hero-copy">
            <div className="eyebrow"><Sparkles size={14} /> LLM POST-TRAINING · FIELD GUIDE 2026</div>
            <h1>
              让学生在自己的错误上，
              <span>听懂老师。</span>
            </h1>
            <p className="hero-lede">
              一份从零开始、但不止于“入门”的 On-Policy Distillation 交互教程。
              从 KL 直觉一路走到 verl 源码与 2026 方法谱系。
            </p>
            <div className="hero-actions">
              <Link className="button button-primary" href="/docs">
                开始学习 <ArrowRight size={17} />
              </Link>
              <a className="button button-ghost" href="https://github.com/Palind-Rome/opd-learning-lab" target="_blank" rel="noreferrer">
                <Braces size={17} /> 查看源码
              </a>
            </div>
            <div className="hero-proof" aria-label="课程特点">
              <span><Check size={14} /> 一手论文核验</span>
              <span><Check size={14} /> 源码逐行追踪</span>
              <span><Check size={14} /> 交互式公式</span>
            </div>
          </div>
          <HeroConsole />
        </div>
        <div className="source-ribbon" aria-label="核心资料来源">
          <div className="home-shell ribbon-track">
            <span className="ribbon-label"><CircleDot size={14} /> EVIDENCE BASE</span>
            {sources.map((source) => <span key={source}>{source}</span>)}
          </div>
        </div>
      </section>

      <section className="home-section mental-section">
        <div className="home-shell split-heading">
          <div>
            <p className="section-kicker">ONE MENTAL MODEL</p>
            <h2>OPD 到底在做什么？</h2>
          </div>
          <p>
            它不是把老师生成的标准答案反复喂给学生。学生先按当前策略生成回答，老师再站到学生真实到达的每个前缀上，告诉它下一步的分布应该怎样移动。
          </p>
        </div>
        <div className="home-shell mental-flow">
          <article>
            <span>1</span>
            <div className="flow-symbol">π<sub>θ</sub></div>
            <h3>学生先走</h3>
            <p>从当前学生策略采样 rollout；错误和犹豫都保留下来。</p>
          </article>
          <i aria-hidden="true"><ArrowRight /></i>
          <article>
            <span>2</span>
            <div className="flow-symbol">π<sub>T</sub></div>
            <h3>老师就地点评</h3>
            <p>在同一条学生前缀上查询 teacher log-prob 或局部分布。</p>
          </article>
          <i aria-hidden="true"><ArrowRight /></i>
          <article>
            <span>3</span>
            <div className="flow-symbol">∇θ</div>
            <h3>学生微调方向</h3>
            <p>用逐 token 的稠密信号更新，而不是只等最终答案给一分。</p>
          </article>
        </div>
      </section>

      <section className="home-section curriculum-section">
        <div className="home-shell">
          <div className="section-heading centered">
            <p className="section-kicker">A COURSE, NOT A LINK DUMP</p>
            <h2>四段路，把概念变成工程判断</h2>
            <p>每一章都回答一个具体问题：我在学什么、代码在哪里、怎么知道它真的工作。</p>
          </div>
          <div className="chapter-grid">
            {chapters.map(({ icon: Icon, ...chapter }) => (
              <Link key={chapter.number} href={chapter.href} className="chapter-card" data-tone={chapter.tone}>
                <div className="chapter-top"><span>{chapter.number}</span><Icon size={21} /></div>
                <h3>{chapter.title}</h3>
                <p>{chapter.text}</p>
                <strong>进入章节 <ArrowRight size={15} /></strong>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="home-section path-section">
        <div className="home-shell path-layout">
          <div className="path-intro">
            <p className="section-kicker">CHOOSE YOUR ROUTE</p>
            <h2>你不需要按目录硬啃。</h2>
            <p>告诉我你现在更像哪一种学习者，得到一条可执行的阅读路线。随时可以切换，不会锁住内容。</p>
          </div>
          <PathSelector />
        </div>
      </section>

      <section className="home-section evidence-section">
        <div className="home-shell evidence-card">
          <div>
            <span className="evidence-stamp">SOURCE<br />AUDITED</span>
          </div>
          <div>
            <p className="section-kicker">WHAT WE PROMISE</p>
            <h2>把“论文说了什么”和“代码真的做了什么”分开。</h2>
            <p>
              课程优先引用论文、作者仓库与官方文档。实验结果标注为“作者报告”；没有官方实现就明确写“未核验到源码”；推断不会伪装成结论。
            </p>
          </div>
          <Link className="button button-light" href="/docs/frontier/reading-list">
            查看证据与必读清单 <ArrowRight size={17} />
          </Link>
        </div>
      </section>

      <footer className="home-footer">
        <div className="home-shell">
          <span>OPD Learning Lab</span>
          <p>Learn the state distribution. Read the loss. Trace the code.</p>
          <Link href="/docs">打开课程 <ArrowRight size={14} /></Link>
        </div>
      </footer>
    </main>
  );
}
