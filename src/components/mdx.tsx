import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import {
  CompatibilityLab,
  GranularityLab,
  KLDivergenceLab,
  KnowledgeCheck,
  OPDCycle,
} from '@/components/learning-labs';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    CompatibilityLab,
    GranularityLab,
    KLDivergenceLab,
    KnowledgeCheck,
    OPDCycle,
    Tab,
    Tabs,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
