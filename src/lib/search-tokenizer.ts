import type { Tokenizer } from '@orama/orama';

const hanRun = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/gu;
const latinToken = /[a-z0-9]+/g;

/**
 * Deterministic mixed Chinese/English tokenizer shared by build-time indexing
 * and the browser search database. Chinese runs are indexed as unigrams and
 * bigrams so both short terms ("熵") and phrases ("知识蒸馏") remain searchable.
 */
export function createSearchTokenizer(): Tokenizer {
  const normalizationCache = new Map<string, string>();

  return {
    language: 'opd-zh',
    normalizationCache,
    tokenize(raw: string) {
      let normalized = normalizationCache.get(raw);
      if (!normalized) {
        normalized = raw.normalize('NFKC').toLowerCase();
        normalizationCache.set(raw, normalized);
      }

      const tokens = new Set<string>();

      for (const token of normalized.match(latinToken) ?? []) tokens.add(token);

      for (const match of normalized.matchAll(hanRun)) {
        const characters = Array.from(match[0]);
        for (const character of characters) tokens.add(character);
        for (let index = 0; index < characters.length - 1; index += 1) {
          tokens.add(characters[index] + characters[index + 1]);
        }
      }

      return [...tokens];
    },
  };
}
