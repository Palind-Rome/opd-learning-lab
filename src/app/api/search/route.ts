import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';
import { createSearchTokenizer } from '@/lib/search-tokenizer';

export const revalidate = false;

export const { staticGET: GET } = createFromSource(source, {
  tokenizer: createSearchTokenizer(),
  search: {
    threshold: 0,
    tolerance: 0,
  },
});
