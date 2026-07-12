import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { appName, gitConfig } from './shared';
import { BrandMark } from '@/components/brand-mark';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="flex items-center gap-2.5 font-semibold tracking-tight">
          <BrandMark />
          <span>{appName}</span>
        </span>
      ),
      transparentMode: 'top',
    },
    links: [
      { text: '学习路径', url: '/docs', active: 'nested-url' },
      { text: '源码地图', url: '/docs/engineering/verl-architecture', active: 'nested-url' },
      { text: '论文索引', url: '/docs/frontier/reading-list', active: 'nested-url' },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
