import { Selection } from './selections';

export const coerceNumber = (value: unknown, fallback = 0): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const formatPercentSafe = (
  value: number | null | undefined,
  digits = 2,
  emptyText = '--',
): string => {
  if (value === null || value === undefined) return emptyText;
  if (!Number.isFinite(value)) return emptyText;
  return `${(value * 100).toFixed(digits)}%`;
};

export type SeriesPoint = {

  timestamp: string;

  value: number;

};



export type SeriesContributor = {

  contributor_id: string;

  label: string;

  symbol?: string | null;

  interval?: string | null;

  strategy?: string | null;

  points: SeriesPoint[];

};



export type SeriesResponse = {

  series: string;

  selection?: Selection;

  downsampled?: boolean;

  raw_count?: number;

  downsampled_count?: number;

  portfolio: SeriesPoint[];

  perFile: SeriesContributor[];

};



type RawSeriesResponse = Omit<SeriesResponse, 'perFile'> & {

  per_file?: SeriesContributor[];

  perFile?: SeriesContributor[];

};



export type HistogramBucket = {

  bucket: string;

  count: number;

};



export type HistogramResponse = {

  label: string;

  buckets: HistogramBucket[];

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



export type CorrelationResponse = {

  labels: string[];

  matrix: number[][];

  mode: string;

  notes: string[];

};



export type FileMetadata = {

  file_id: string;

  filename: string;

  symbols: string[];

  intervals: string[];

  strategies: string[];

  date_min?: string | null;

  date_max?: string | null;

  mtm_available: boolean;

  margin_per_contract?: number | null;

  big_point_value?: number | null;

};



export type FileUploadResponse = {

  job_id: string;

  files: FileMetadata[];

  message?: string;

};



export type SelectionMeta = {
  symbols: string[];
  intervals: string[];
  strategies: string[];
  date_min?: string | null;
  date_max?: string | null;
  files: FileMetadata[];
  account_equity: number;
};


export type AllocationRow = {
  asset: string;
  label: string;
  weight: number;
  contracts: number;
  margin_per_contract?: number | null;
};

export type ContractRow = {
  asset: string;
  contracts: number;
  notional: number;
  margin?: number | null;
};

export type OptimizerSummary = {
  objective: string;
  risk_measure: string;
  expected_return: number;
  risk: number;
  sharpe: number;
  max_drawdown: number;
  capital: number;
};

export type FrontierPoint = {
  expected_return: number;
  risk: number;
  weights: Record<string, number>;
};

export type JobResult = {
  summary?: OptimizerSummary | null;
  weights: AllocationRow[];
  contracts: ContractRow[];
  frontier: FrontierPoint[];
};

export type OptimizerJobResponse = {
  job_id: string;
  status: string;
  message: string;
};

export type JobStatusResponse = {
  job_id: string;
  status: string;
  progress: number;
  result?: JobResult | null;
  error?: string | null;
};

export type MeanRiskPayload = {
  objective: string;
  risk_measure: string;
  return_model: string;
  method_mu: string;
  method_cov: string;
  solver?: string | null;
  risk_free_rate: number;
  risk_aversion: number;
  alpha: number;
  budget: number;
  bounds: {
    default_min: number;
    default_max: number;
    overrides: Record<string, [number | null, number | null]>;
  };
  symbol_caps: { name: string; max_weight: number }[];
  strategy_caps: { name: string; max_weight: number }[];
  efficient_frontier_points: number;
  max_risk?: number | null;
  min_return?: number | null;
  turnover_limit?: number | null;
};



type ApiSelectionPayload = {
  files: string[];
  symbols: string[];
  intervals: string[];
  strategies: string[];
  direction?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  contract_multipliers: Record<string, number>;
  margin_overrides: Record<string, number>;
  spike_flag: boolean;
  account_equity?: number | null;
};

const selectionToApiPayload = (selection: Selection): ApiSelectionPayload => {
  const direction =
    selection.direction && selection.direction.toLowerCase() !== 'all'
      ? selection.direction
      : null;
  const contractMultipliers = selection.contractMultipliers ?? selection.contracts ?? {};
  const marginOverrides = selection.marginOverrides ?? selection.margins ?? {};
  return {
    files: selection.files,
    symbols: selection.symbols,
    intervals: selection.intervals,
    strategies: selection.strategies,
    direction,
    start_date: selection.start ?? null,
    end_date: selection.end ?? null,
    contract_multipliers: contractMultipliers,
    margin_overrides: marginOverrides,
    spike_flag: Boolean(selection.spike),
    account_equity: selection.accountEquity ?? null,
  };
};


export type CTAResponse = {

  summary: {

    roi: number;

    annualRor: number;

    timeInMarket: number;

    longestFlat: number;

    maxRunup: number;

    maxDrawdown: number;

    avgMonthly: number;

    stdMonthly: number;

  };

  monthly: { month: string; additive: number; compounded: number }[];

  annual: { year: number; additive: number; compounded: number }[];

};



const API_BASE = process.env.NEXT_PUBLIC_API_BASE;



export type SeriesKind =

  | 'equity'

  | 'drawdown'

  | 'equityPercent'

  | 'intradayDrawdown'

  | 'netpos'

  | 'margin';



const endpoint: Record<SeriesKind | 'metrics' | 'histogram', string> = {

  equity: '/api/v1/series/equity',

  drawdown: '/api/v1/series/drawdown',

  equityPercent: '/api/v1/series/equity-percent',

  intradayDrawdown: '/api/v1/series/intraday-dd',

  netpos: '/api/v1/series/netpos',

  margin: '/api/v1/series/margin',

  histogram: '/api/v1/series/histogram',

  metrics: '/api/v1/metrics',

};

const correlationEndpoint = '/api/v1/correlations';

const ctaEndpoint = '/api/v1/cta';

const riskfolioEndpoint = '/api/v1/optimizer/riskfolio';

const jobsEndpoint = '/api/v1/jobs';

const filesEndpoint = '/api/v1/files';

const selectionMetaEndpoint = '/api/v1/selection/meta';

const uploadEndpoint = '/api/v1/upload';



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



const NO_DATA_ERROR_CODE = 'NO_DATA';

const NO_DATA_ERROR_MESSAGE = 'No data matches your selection';



type CodedError = Error & { code?: string };



function raiseResponseError(response: Response): never {

  if (response.status === 404) {

    const error = new Error(NO_DATA_ERROR_MESSAGE) as CodedError;

    error.code = NO_DATA_ERROR_CODE;

    throw error;

  }

  throw new Error(`Request failed: ${response.status}`);

}



function selectionQuery(selection: Selection, opts?: { downsample?: boolean }) {
  const params = new URLSearchParams();
  selection.files.forEach((file) => params.append('files', file));
  selection.symbols.forEach((symbol) => params.append('symbols', symbol));

  selection.intervals.forEach((interval) => params.append('intervals', interval));

  selection.strategies.forEach((strategy) => params.append('strategies', strategy));

  if (selection.direction) params.set('direction', selection.direction);

  if (selection.start) params.set('start_date', selection.start);

  if (selection.end) params.set('end_date', selection.end);

  const contractMultipliers = selection.contractMultipliers ?? {};

  const marginOverrides = selection.marginOverrides ?? {};

  Object.entries(contractMultipliers).forEach(([key, val]) => params.append('contract_multipliers', `${key}:${val}`));
  Object.entries(marginOverrides).forEach(([key, val]) => params.append('margin_overrides', `${key}:${val}`));
  params.set('spike_flag', String(selection.spike));
  if (selection.accountEquity !== undefined && selection.accountEquity !== null) {
    params.set('account_equity', String(selection.accountEquity));
  }
  if (opts && opts.downsample !== undefined) {
    params.set('downsample', String(opts.downsample));
  }
  return params.toString();

}



export function mockSeries(selection: Selection, kind: SeriesKind): SeriesResponse {

  const rand = seeded(`${selection.name}-${kind}`);

  const points: SeriesPoint[] = [];

  let running = 0;

  const start = Date.now() - 1000 * 60 * 60 * 24 * 120;

  const days = 90;

  for (let i = 0; i < days; i += 1) {

    const drift = (rand() - 0.48) * 1200;

    running += drift;

    const base = kind === 'drawdown' || kind === 'intradayDrawdown' ? -Math.abs(running * 0.6) : 25000;

    const value = (() => {

      if (kind === 'drawdown' || kind === 'intradayDrawdown') return Math.min(-Math.abs(running * 0.75), -100);

      if (kind === 'equityPercent') return 100 + running / 1000;

      if (kind === 'netpos') return Math.round((rand() - 0.5) * 8);

      if (kind === 'margin') return Math.max(0, 5000 + running * 0.15 + rand() * 800);

      return base + running;

    })();

    points.push({

      timestamp: new Date(start + i * 86400000).toISOString().slice(0, 10),

      value,

    });

  }

  const labelByKind: Record<SeriesKind, string> = {

    equity: 'Equity Curve',

    drawdown: 'Drawdown',

    equityPercent: 'Percent Equity',

    intradayDrawdown: 'Intraday Drawdown',

    netpos: 'Net Position',

    margin: 'Margin Usage',

  };

  const perFile = selection.files.map((id, idx) => {

    const label = selection.fileLabels?.[id] || id;

    const modifier = 1 + idx * 0.05;

    return {

      contributor_id: id,

      label,

      points: points.map((pt) => ({ ...pt, value: pt.value * modifier })),

      symbol: undefined,

      interval: undefined,

      strategy: undefined,

    };

  });

  return {

    series: `${labelByKind[kind]} (${selection.name})`,

    portfolio: points,

    perFile,

    downsampled: false,

    raw_count: points.length,

    downsampled_count: points.length,

  };

}



export function mockHistogram(selection: Selection): HistogramResponse {

  const rand = seeded(`${selection.name}-histogram`);

  const buckets: HistogramBucket[] = [];

  for (let i = -6; i <= 8; i += 1) {

    buckets.push({ bucket: `${i * 10}%`, count: Math.max(1, Math.round(rand() * 120)) });

  }

  return { label: `Return distribution (${selection.name})`, buckets };

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



export function mockCorrelations(selection: Selection, mode: string = 'drawdown_pct'): CorrelationResponse {

  const labels = selection.files.length ? selection.files : ['Portfolio'];

  const rand = seeded(`${selection.name}-corr`);

  const size = labels.length || 2;

  const matrix: number[][] = Array.from({ length: size }, (_, row) =>

    Array.from({ length: size }, (_, col) => {

      if (row === col) return 1;

      const base = rand() * 0.7;

      return parseFloat(((rand() > 0.5 ? base : -base)).toFixed(3));

    }),

  );

  return {

    labels,

    matrix,

    mode,

    notes: [

      'Mirrors legacy correlation heatmap (drawdown %, returns, P/L, slope modes).',

      'Mocked values fall between -0.7 and 0.7 to show diversification effects.',

    ],

  };

}



export function mockCta(selection: Selection): CTAResponse {

  const rand = seeded(`${selection.name}-cta`);

  const months: CTAResponse['monthly'] = [];

  for (let i = 1; i <= 12; i += 1) {

    const additive = parseFloat(((rand() - 0.45) * 10).toFixed(2));

    const compounded = parseFloat((additive - 0.4 + rand()).toFixed(2));

    months.push({ month: `2024-${String(i).padStart(2, '0')}`, additive, compounded });

  }

  const annual: CTAResponse['annual'] = [

    { year: 2023, additive: 12.1, compounded: 10.4 },

    { year: 2024, additive: 16.4, compounded: 15.1 },

  ];

  return {

    summary: {

      roi: 34.2,

      annualRor: 16.4,

      timeInMarket: 62.5,

      longestFlat: 42,

      maxRunup: 18.7,

      maxDrawdown: -12.6,

      avgMonthly: 1.8,

      stdMonthly: 3.1,

    },

    monthly: months,

    annual,

  };

}



export async function fetchSeries(selection: Selection, kind: SeriesKind, downsample: boolean = true) {

  const query = selectionQuery(selection, { downsample });

  const path = `${endpoint[kind]}?${query}`;

  if (!API_BASE) {

    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot load series data');

  }

  const response = await fetch(`${API_BASE}${path}`, { cache: 'no-store' });

  if (!response.ok) {

    raiseResponseError(response);

  }

  const payload = (await response.json()) as RawSeriesResponse;

  const { per_file, perFile, ...rest } = payload as RawSeriesResponse & { per_file?: SeriesContributor[] };

  return {

    ...rest,

    perFile: perFile ?? per_file ?? [],

  };

}



export async function fetchMetrics(selection: Selection) {

  const query = selectionQuery(selection);

  const path = `${endpoint.metrics}?${query}`;

  if (!API_BASE) {

    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot load metrics');

  }

  const response = await fetch(`${API_BASE}${path}`, { cache: 'no-store' });

  if (!response.ok) {

    raiseResponseError(response);

  }

  return (await response.json()) as MetricsRow[];

}



export async function fetchHistogram(selection: Selection) {

  const query = selectionQuery(selection);

  const path = `${endpoint.histogram}?${query}`;

  if (!API_BASE) {

    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot load histogram');

  }

  const response = await fetch(`${API_BASE}${path}`, { cache: 'no-store' });

  if (!response.ok) {

    raiseResponseError(response);

  }

  return (await response.json()) as HistogramResponse;

}



export async function fetchCorrelations(selection: Selection, mode: string = 'drawdown_pct') {

  const query = `${selectionQuery(selection)}&mode=${mode}`;

  const fallback = mockCorrelations(selection, mode);

  const path = `${correlationEndpoint}?${query}`;

  return fetchJson<CorrelationResponse>(path, fallback);

}



export async function fetchCta(selection: Selection) {

  const query = selectionQuery(selection);

  const fallback = mockCta(selection);

  const path = `${ctaEndpoint}?${query}`;

  return fetchJson<CTAResponse>(path, fallback);

}



export async function submitRiskfolioJob(selection: Selection, meanRisk: MeanRiskPayload) {
  if (!API_BASE) {
    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot run optimizer');
  }
  const payload = {
    selection: selectionToApiPayload(selection),
    constraints: {},
    objective: 'mean_risk',
    tags: [],
    mean_risk: meanRisk,
  };
  const response = await fetch(`${API_BASE}${riskfolioEndpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as OptimizerJobResponse;
}

export async function fetchJobStatus(jobId: string) {
  if (!API_BASE) {
    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot load job status');
  }
  const response = await fetch(`${API_BASE}${jobsEndpoint}/${jobId}`, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as JobStatusResponse;
}



export async function listFiles(): Promise<FileMetadata[]> {

  if (!API_BASE) {

    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot load files');

  }

  const response = await fetch(`${API_BASE}${filesEndpoint}`, { cache: 'no-store' });

  if (!response.ok) {

    throw new Error(`Request failed: ${response.status}`);

  }

  return (await response.json()) as FileMetadata[];

}



export async function getSelectionMeta(): Promise<SelectionMeta> {

  if (!API_BASE) {

    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot load selection metadata');

  }

  const response = await fetch(`${API_BASE}${selectionMetaEndpoint}`, { cache: 'no-store' });

  if (!response.ok) {

    throw new Error(`Request failed: ${response.status}`);

  }

  return (await response.json()) as SelectionMeta;

}



export async function uploadFiles(formData: FormData): Promise<FileUploadResponse> {

  if (!API_BASE) {

    throw new Error('NEXT_PUBLIC_API_BASE is not set; cannot upload files');

  }

  const response = await fetch(`${API_BASE}${uploadEndpoint}`, {

    method: 'POST',

    body: formData,

  });

  if (!response.ok) {

    throw new Error(`Upload failed: ${response.status}`);

  }

  return (await response.json()) as FileUploadResponse;

}

