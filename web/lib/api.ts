import type { Selection } from './types/selection';

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

  start_value: number;

  end_value: number;

};



export type HistogramResponse = {

  label: string;

  buckets: HistogramBucket[];

};



export type MetricsBlock = {
  key: string;
  label: string;
  fileId?: string | null;
  metrics: Record<string, number | string | null>;
};

export type MetricsResponse = {
  selection: Selection;
  portfolio: MetricsBlock;
  files: MetricsBlock[];
};



export type CorrelationResponse = {
  labels: string[];
  matrix: number[][];
  mode: string;
  notes?: string[];
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

export type BacktestSeriesLine = {
  label: string;
  points: SeriesPoint[];
};

export type OptimizerCorrelation = {
  mode: string;
  labels: string[];
  matrix: number[][];
};

export type JobResult = {
  summary?: OptimizerSummary | null;
  weights: AllocationRow[];
  contracts: ContractRow[];
  frontier: FrontierPoint[];
  backtest: BacktestSeriesLine[];
  correlation?: OptimizerCorrelation | null;
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


const API_BASE = process.env.NEXT_PUBLIC_API_BASE;



export type SeriesKind =

  | 'equity'

  | 'drawdown'

  | 'equityPercent'

  | 'intradayDrawdown'

  | 'netpos'

  | 'margin';

export type ExposureView =
  | 'portfolio_daily'
  | 'portfolio_step'
  | 'per_symbol'
  | 'per_file';



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

const riskfolioEndpoint = '/api/v1/optimizer/riskfolio';

const jobsEndpoint = '/api/v1/jobs';

const filesEndpoint = '/api/v1/files';

const selectionMetaEndpoint = '/api/v1/selection/meta';

const uploadEndpoint = '/api/v1/upload';



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



function selectionQuery(selection: Selection, opts?: { downsample?: boolean; exposureView?: ExposureView }) {
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
  if (opts?.exposureView) {
    params.set('exposure_view', opts.exposureView);
  }
  return params.toString();

}



export async function fetchSeries(
  selection: Selection,
  kind: SeriesKind,
  opts?: { downsample?: boolean; exposureView?: ExposureView },
) {
  const query = selectionQuery(selection, { downsample: opts?.downsample ?? true, exposureView: opts?.exposureView });

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

  return (await response.json()) as MetricsResponse;

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
