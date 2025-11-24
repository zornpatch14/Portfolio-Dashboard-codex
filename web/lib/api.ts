import { Selection } from './selections';

export type SeriesPoint = {
  timestamp: string;
  value: number;
};

export type SeriesResponse = {
  label: string;
  points: SeriesPoint[];
  rawCount?: number;
  downsampledCount?: number;
};

export type MetricsRow = {
  scope: string;
  netProfit: number;
  drawdown: number;
  trades: number;
  winRate: number;
  expectancy: number;
  exposure: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

const endpoint = {
  equity: '/api/v1/series/equity',
  drawdown: '/api/v1/series/drawdown',
  metrics: '/api/v1/metrics',
};

async function fetchJson<T>(path: string, fallback: T): Promise<T> {
  if (!API_BASE) return fallback;
  try {
    const response = await fetch(`${API_BASE}${path}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as T;
  } catch (error) {
    console.warn(`[mock-fallback] ${path}`, error);
    return fallback;
  }
}

function seeded(seed: string) {
  let h = 0;
  for (const ch of seed) {
    h = (h << 5) - h + ch.charCodeAt(0);
    h |= 0;
  }
  return () => {
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    return Math.abs(h) / 0x7fffffff;
  };
}

function selectionQuery(selection: Selection) {
  const params = new URLSearchParams();
  selection.files.forEach((file) => params.append('files', file));
  selection.symbols.forEach((symbol) => params.append('symbols', symbol));
  selection.intervals.forEach((interval) => params.append('intervals', interval));
  selection.strategies.forEach((strategy) => params.append('strategies', strategy));
  if (selection.direction) params.set('direction', selection.direction);
  if (selection.start) params.set('start', selection.start);
  if (selection.end) params.set('end', selection.end);
  Object.entries(selection.contracts).forEach(([key, val]) => params.append(`contracts[${key}]`, String(val)));
  Object.entries(selection.margins).forEach(([key, val]) => params.append(`margins[${key}]`, String(val)));
  params.set('spike', String(selection.spike));
  return params.toString();
}

export function mockSeries(selection: Selection, kind: 'equity' | 'drawdown'): SeriesResponse {
  const rand = seeded(`${selection.name}-${kind}`);
  const points: SeriesPoint[] = [];
  let running = kind === 'equity' ? 0 : 0;
  const start = Date.now() - 1000 * 60 * 60 * 24 * 60;
  for (let i = 0; i < 60; i += 1) {
    const drift = (rand() - 0.45) * (kind === 'equity' ? 1200 : 400);
    running += drift;
    const base = kind === 'equity' ? 25000 : 0;
    points.push({
      timestamp: new Date(start + i * 86400000).toISOString().slice(0, 10),
      value: kind === 'equity' ? base + running : Math.min(-Math.abs(running), 0),
    });
  }
  return {
    label: `${kind === 'equity' ? 'Equity Curve' : 'Drawdown'} (${selection.name})`,
    points,
    rawCount: points.length,
    downsampledCount: points.length,
  };
}

export function mockMetrics(selection: Selection): MetricsRow[] {
  const rand = seeded(selection.name);
  const files = selection.files.length ? selection.files : ['portfolio'];
  return files.map((file, idx) => {
    const base = rand();
    const trades = Math.round(40 + base * 120);
    return {
      scope: idx === files.length - 1 ? `${file} (portfolio)` : file,
      netProfit: Math.round(12000 + base * 15000),
      drawdown: Math.round(-2000 - base * 3000),
      trades,
      winRate: Math.round(45 + base * 35),
      expectancy: parseFloat((50 + base * 150).toFixed(2)),
      exposure: parseFloat((0.8 + base * 1.6).toFixed(2)),
    };
  });
}

export async function fetchSeries(selection: Selection, kind: 'equity' | 'drawdown') {
  const query = selectionQuery(selection);
  const fallback = mockSeries(selection, kind);
  const path = `${endpoint[kind]}?${query}`;
  return fetchJson<SeriesResponse>(path, fallback);
}

export async function fetchMetrics(selection: Selection) {
  const query = selectionQuery(selection);
  const fallback = mockMetrics(selection);
  const path = `${endpoint.metrics}?${query}`;
  return fetchJson<MetricsRow[]>(path, fallback);
}
