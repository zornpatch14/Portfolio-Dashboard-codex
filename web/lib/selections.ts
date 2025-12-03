import selectionsData from '../data/selections.json';

export type Selection = {
  name: string;
  files: string[];
  fileLabels?: Record<string, string>;
  symbols: string[];
  intervals: string[];
  strategies: string[];
  direction: string;
  start: string | null;
  end: string | null;
  contracts: Record<string, number>;
  margins: Record<string, number>;
  contractMultipliers?: Record<string, number>;
  marginOverrides?: Record<string, number>;
  spike: boolean;
  downsample?: boolean;
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

function sortNumericStrings(values: string[]): string[] {
  return [...values].sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    const aNaN = Number.isNaN(na);
    const bNaN = Number.isNaN(nb);
    if (!aNaN && !bNaN) {
      return na - nb;
    }
    if (aNaN && bNaN) {
      return a.localeCompare(b);
    }
    return aNaN ? 1 : -1;
  });
}

function sortMap<T>(map: Record<string, T> | undefined): Record<string, T> | undefined {
  if (!map) return map;
  const entries = Object.entries(map);
  if (!entries.length) return {};
  return Object.fromEntries(entries.sort(([a], [b]) => a.localeCompare(b, undefined, { sensitivity: 'base' })));
}

export function normalizeSelection(selection: Selection): Selection {
  const normalizedContracts = sortMap(selection.contractMultipliers ?? selection.contracts) || {};
  const normalizedMargins = sortMap(selection.marginOverrides ?? selection.margins) || {};
  return {
    ...selection,
    files: [...(selection.files || [])].sort(),
    symbols: [...(selection.symbols || [])].map((symbol) => symbol.toUpperCase()).sort(),
    intervals: sortNumericStrings(selection.intervals || []),
    strategies: [...(selection.strategies || [])].sort(),
    contracts: normalizedContracts,
    margins: normalizedMargins,
    contractMultipliers: normalizedContracts,
    marginOverrides: normalizedMargins,
  };
}
