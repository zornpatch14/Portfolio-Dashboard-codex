import selectionsData from '../data/selections.json';

export type Selection = {
  name: string;
  files: string[];
  symbols: string[];
  intervals: string[];
  strategies: string[];
  direction: string;
  start: string | null;
  end: string | null;
  contracts: Record<string, number>;
  margins: Record<string, number>;
  spike: boolean;
};

export function loadSampleSelections(): Selection[] {
  // Mirrored from tests/baseline/selections.json to keep the UI aligned with parity fixtures.
  return selectionsData as Selection[];
}

export function selectionLabel(selection: Selection): string {
  const symbols = selection.symbols.length ? selection.symbols.join(', ') : 'All symbols';
  const intervals = selection.intervals.length ? selection.intervals.join(', ') : 'All intervals';
  const strategies = selection.strategies.length ? selection.strategies.join(', ') : 'All strategies';
  return `${symbols} | ${intervals} | ${strategies}`;
}
