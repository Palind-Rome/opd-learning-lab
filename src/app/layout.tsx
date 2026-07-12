import { Provider } from '@/components/provider';
import type { Metadata } from 'next';
import 'katex/dist/katex.css';
import './global.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://palind-rome.github.io/opd-learning-lab/'),
  title: {
    default: 'OPD Learning Lab',
    template: '%s · OPD Learning Lab',
  },
  description: '从直觉、公式到 verl 源码，系统学习大语言模型 On-Policy Distillation。',
  keywords: ['OPD', 'On-Policy Distillation', 'LLM post-training', '知识蒸馏', 'verl'],
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
