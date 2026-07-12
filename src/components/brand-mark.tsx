export function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="brand-mark" aria-hidden="true" data-compact={compact || undefined}>
      <span>π</span>
      <i />
    </span>
  );
}
