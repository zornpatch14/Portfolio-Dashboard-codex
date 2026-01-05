'use client';



import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useQuery } from '@tanstack/react-query';

import MetricsGrid from '../components/MetricsGrid';

import SeriesChart from '../components/SeriesChart';
import { HistogramChart } from '../components/HistogramChart';
import { SelectionControls } from '../components/SelectionControls';
import { CorrelationHeatmap } from '../components/CorrelationHeatmap';
import { EquityMultiChart } from '../components/EquityMultiChart';
import { EfficientFrontierChart } from '../components/EfficientFrontierChart';
import { FrontierAllocationAreaChart } from '../components/FrontierAllocationAreaChart';

import {
  fetchHistogram,
  fetchMetrics,
  fetchSeries,
  fetchJobStatus,
  submitRiskfolioJob,
  uploadFiles,
  getSelectionMeta,
  listFiles,
  SeriesResponse,
  JobStatusResponse,
  MeanRiskPayload,
  coerceNumber,
  SelectionMeta,
} from '../lib/api';

import { blankSelection, normalizeSelection } from '../lib/selections';
import type { Selection } from '../lib/types/selection';



const SELECTION_STORAGE_KEY = 'portfolio-selection-state';

const UPLOAD_INPUT_ID = 'upload-input';

const MARGIN_DEFAULTS: Record<string, number> = {
  MNQ: 3250,
  MES: 2300,
  MYM: 1500,
  M2K: 1000,
  CD: 1100,
  JY: 3100,
  NE1: 1500,
  NG: 3800,
};
const FALLBACK_MARGIN = 10000;
const ACCOUNT_EQUITY_FALLBACK = 25000;

const tabs = [
  { key: 'load-trade-lists', label: 'Load Trade Lists' },

  { key: 'summary', label: 'Summary' },

  { key: 'equity-curves', label: 'Equity Curves' },

  { key: 'portfolio-drawdown', label: 'Portfolio Drawdown' },

  { key: 'margin', label: 'Margin' },

  { key: 'trade-pl-histogram', label: 'Trade P/L Histogram' },

  { key: 'correlations', label: 'Correlations' },

  { key: 'riskfolio', label: 'Riskfolio' },

  { key: 'cta-report', label: 'CTA Report' },

  { key: 'metrics', label: 'Metrics' },

] as const;

const NO_DATA_MESSAGE = 'No data matches your selection.';

const GENERIC_ERROR_MESSAGE = 'Failed to load data.';

const STALE_TIME = Infinity;



const formatPercent = (value: number | null | undefined, digits = 2) => {
  if (value === undefined || value === null || Number.isNaN(value)) return '--';
  return `${(value * 100).toFixed(digits)}%`;
};

const formatCurrency = (value: number | null | undefined) => {
  if (value === undefined || value === null || Number.isNaN(value)) return '--';
  return `$${value.toLocaleString()}`;
};

const formatHistogramDollar = (value: number) =>
  `$${value.toLocaleString(undefined, { maximumFractionDigits: 0, minimumFractionDigits: 0 })}`;

const formatHistogramDollarRange = (start: number, end: number) =>
  `${formatHistogramDollar(start)} to ${formatHistogramDollar(end)}`;

const formatHistogramPercentRange = (start: number, end: number, equity: number) => {
  if (!equity) return '0.0% to 0.0%';
  const startPct = (start / equity) * 100;
  const endPct = (end / equity) * 100;
  return `${startPct.toFixed(1)}% to ${endPct.toFixed(1)}%`;
};

const parseOptionalNumber = (raw: string): number | null => {
  if (!raw.trim()) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
};

const parseCapsInput = (input: string): { name: string; max_weight: number }[] =>
  input
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [rawName, rawValue] = entry.split('=');
      const name = (rawName || '').trim();
      const value = Number((rawValue || '').trim());
      if (!name || !Number.isFinite(value)) {
        return null;
      }
      return { name: name.toUpperCase(), max_weight: value };
    })
    .filter((entry): entry is { name: string; max_weight: number } => entry !== null);
type BoundsOverrideState = Record<string, { min?: string; max?: string }>;

const clampToUnitInterval = (value: number) => Math.min(1, Math.max(0, value));

const sanitizeWeightInputString = (value: string): string => {
  if (value === '') return '';
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return value;
  const clamped = clampToUnitInterval(parsed);
  return clamped !== parsed ? clamped.toString() : value;
};

const enforceWeightInputBounds = (value: string, fallback: string): string => {
  if (!value.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return clampToUnitInterval(parsed).toString();
};

const parseWeightInputValue = (value: string, fallback: number): number => {
  if (!value.trim()) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return clampToUnitInterval(parsed);
};

const parseOptionalWeightInputValue = (value?: string | null): number | null => {
  if (!value || !value.trim()) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  return clampToUnitInterval(parsed);
};



type FilterValueMap = {

  symbols: string[];

  intervals: string[];

  strategies: string[];

};

type SelectionDefaults = Pick<
  Selection,
  | 'fileLabels'
  | 'files'
  | 'symbols'
  | 'intervals'
  | 'strategies'
  | 'direction'
  | 'spike'
  | 'contracts'
  | 'contractMultipliers'
  | 'margins'
  | 'marginOverrides'
  | 'start'
  | 'end'
>;

const deriveSelectionDefaults = (meta: SelectionMeta, prev: Selection): SelectionDefaults => {
  const files = meta.files.map((file) => file.file_id);
  const fileLabels = Object.fromEntries(meta.files.map((file) => [file.file_id, file.filename]));

  const symbolSet = new Set<string>();
  const intervalSet = new Set<string>();
  const strategySet = new Set<string>();

  meta.files.forEach((file) => {
    file.symbols.forEach((symbol) => {
      if (symbol) symbolSet.add(symbol);
    });
    file.intervals.forEach((interval) => {
      if (interval) intervalSet.add(String(interval));
    });
    file.strategies.forEach((strategy) => {
      if (strategy) strategySet.add(strategy);
    });
  });

  const symbols = Array.from(symbolSet).sort();
  const intervals = Array.from(intervalSet).sort((a, b) => Number(a) - Number(b));
  const strategies = Array.from(strategySet).sort();

  const existingContracts = prev.contractMultipliers ?? prev.contracts ?? {};
  const contracts: Record<string, number> = {};
  files.forEach((fileId) => {
    contracts[fileId] = existingContracts[fileId] ?? 1;
  });

  const inferMarginDefault = (file: SelectionMeta['files'][number]): number => {
    if (typeof file.margin_per_contract === 'number' && file.margin_per_contract > 0) {
      return file.margin_per_contract;
    }
    const symbol = (file.symbols[0] || '').toUpperCase();
    if (symbol && MARGIN_DEFAULTS[symbol]) {
      return MARGIN_DEFAULTS[symbol];
    }
    return FALLBACK_MARGIN;
  };

  const existingMargins = prev.marginOverrides ?? prev.margins ?? {};
  const margins: Record<string, number> = {};
  meta.files.forEach((file) => {
    const fileId = file.file_id;
    if (Object.prototype.hasOwnProperty.call(existingMargins, fileId)) {
      margins[fileId] = existingMargins[fileId];
    } else {
      margins[fileId] = inferMarginDefault(file);
    }
  });

  const dateCandidates = meta.files
    .flatMap((file) => [file.date_min, file.date_max])
    .filter((value): value is string => Boolean(value));
  const derivedStart = meta.date_min ?? dateCandidates.reduce<string | undefined>((acc, value) => {
    if (!acc || value < acc) return value;
    return acc;
  }, undefined);
  const derivedEnd = meta.date_max ?? dateCandidates.reduce<string | undefined>((acc, value) => {
    if (!acc || value > acc) return value;
    return acc;
  }, undefined);

  return {
    fileLabels,
    files,
    symbols,
    intervals,
    strategies,
    direction: 'All',
    spike: true,
    contracts,
    contractMultipliers: contracts,
    margins,
    marginOverrides: margins,
    start: prev.start ?? derivedStart ?? null,
    end: prev.end ?? derivedEnd ?? null,
  };
};

type RiskfolioSuggestedRow = {
  asset: string;
  label: string;
  weight: number;
  marginPerContract: number;
  suggestedContracts: number;
  currentContracts: number;
  suggestedMargin: number;
  currentMargin: number;
  suggestedGap: number | null;
  currentGap: number | null;
  delta: number;
};

type RiskfolioSuggestedSummary = {
  suggestedContractsTotal: number;
  currentContractsTotal: number;
  suggestedMarginTotal: number;
  currentMarginTotal: number;
  maxSuggestedGap: number | null;
  avgSuggestedGap: number | null;
  maxCurrentGap: number | null;
  avgCurrentGap: number | null;
};

type RiskfolioSuggestions = {
  rows: RiskfolioSuggestedRow[];
  summary: RiskfolioSuggestedSummary | null;
  note: string;
};



const arraysEqual = (a: string[] = [], b: string[] = []) => a.length === b.length && a.every((val, idx) => val === b[idx]);



const isNoDataError = (error: unknown): boolean => Boolean(error && (error as any).code === 'NO_DATA');



const getQueryErrorMessage = (error: unknown, fallback?: string) =>

  isNoDataError(error) ? NO_DATA_MESSAGE : fallback || GENERIC_ERROR_MESSAGE;



type TabKey = (typeof tabs)[number]['key'];



export default function HomePage() {

  const [activeSelection, setActiveSelection] = useState<Selection>(blankSelection);

  const [activeTab, setActiveTab] = useState<TabKey>('load-trade-lists');

  const [includeDownsample, setIncludeDownsample] = useState(false);

  const [exportFormat, setExportFormat] = useState<'csv' | 'parquet'>('parquet');

  const [plotEnabled, setPlotEnabled] = useState<Record<string, boolean>>({});

  const [plotDrawdownEnabled, setPlotDrawdownEnabled] = useState<Record<string, boolean>>({});

  const [plotMarginEnabled, setPlotMarginEnabled] = useState<Record<string, boolean>>({});
  const [showExposureDebug, setShowExposureDebug] = useState(false);

  const [plotHistogramEnabled, setPlotHistogramEnabled] = useState<Record<string, boolean>>({});

  const [riskfolioMode, setRiskfolioMode] = useState<'mean-risk' | 'risk-parity' | 'hierarchical'>('mean-risk');
  const [meanRiskObjective, setMeanRiskObjective] = useState('MinRisk');
  const [meanRiskReturnModel, setMeanRiskReturnModel] = useState('arithmetic');
  const [meanRiskReturnEstimate, setMeanRiskReturnEstimate] = useState('hist');
  const [meanRiskRiskMeasure, setMeanRiskRiskMeasure] = useState('CVaR');
  const [meanRiskCovariance, setMeanRiskCovariance] = useState('ledoit');
  const [meanRiskRiskFree, setMeanRiskRiskFree] = useState(5);
  const [meanRiskRiskAversion, setMeanRiskRiskAversion] = useState(2);
  const [meanRiskAlpha, setMeanRiskAlpha] = useState(0.05);
  const [meanRiskMinBound, setMeanRiskMinBound] = useState('0');
  const [meanRiskMaxBound, setMeanRiskMaxBound] = useState('1');
  const [meanRiskBudget, setMeanRiskBudget] = useState(1);
  const [meanRiskSymbolCaps, setMeanRiskSymbolCaps] = useState('');
  const [meanRiskStrategyCaps, setMeanRiskStrategyCaps] = useState('');
  const [meanRiskMaxRisk, setMeanRiskMaxRisk] = useState('');
  const [meanRiskMinReturn, setMeanRiskMinReturn] = useState('');
  const [meanRiskTurnover, setMeanRiskTurnover] = useState('');
  const [meanRiskFrontierPoints, setMeanRiskFrontierPoints] = useState(20);
  const [meanRiskOverrides, setMeanRiskOverrides] = useState<BoundsOverrideState>({});
  const [optimizerJobId, setOptimizerJobId] = useState<string | null>(null);
  const [optimizerStatus, setOptimizerStatus] = useState<JobStatusResponse | null>(null);
  const [optimizerLoading, setOptimizerLoading] = useState(false);
  const [optimizerError, setOptimizerError] = useState<string | null>(null);
  const [riskfolioApplyMessage, setRiskfolioApplyMessage] = useState<string | null>(null);

  const minWeightBound = useMemo(() => parseWeightInputValue(meanRiskMinBound, 0), [meanRiskMinBound]);
  const maxWeightBound = useMemo(() => parseWeightInputValue(meanRiskMaxBound, 1), [meanRiskMaxBound]);

  const [selectionMeta, setSelectionMeta] = useState<Awaited<ReturnType<typeof getSelectionMeta>> | null>(null);
  const [filesMeta, setFilesMeta] = useState<Awaited<ReturnType<typeof listFiles>>>([]);

  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const apiBase = process.env.NEXT_PUBLIC_API_BASE;

  const apiMissing = !apiBase;

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const accountEquity = activeSelection.accountEquity ?? selectionMeta?.account_equity ?? ACCOUNT_EQUITY_FALLBACK;
  const [riskfolioContractEquity, setRiskfolioContractEquity] = useState(accountEquity);
  const [selectedFrontierIdx, setSelectedFrontierIdx] = useState<number | null>(null);
  const [metricsRequested, setMetricsRequested] = useState(false);

  const optimizerResult = optimizerStatus?.result ?? null;
  const optimizerWeights = optimizerResult?.weights ?? [];
  const frontierRows = optimizerResult?.frontier ?? [];
  const lastFrontierSignature = useRef<string | null>(null);

  const optimizerProgress = optimizerStatus?.progress ?? 0;

  const optimizerRunning =

    optimizerJobId !== null &&

    (optimizerStatus === null ||

      (optimizerStatus.status !== 'completed' && optimizerStatus.status !== 'failed'));

  const optimizerBacktestSeries = useMemo(() => {
    if (!optimizerResult?.backtest?.length) return [];
    return optimizerResult.backtest.map((line) => ({
      name: line.label,
      points: line.points.map((point) => ({ timestamp: point.timestamp, value: point.value })),
    }));
  }, [optimizerResult]);

  const optimizerCorrelationData = useMemo(() => {
    const payload = optimizerResult?.correlation;
    if (!payload || !payload.matrix?.length || payload.labels.length < 2) {
      return null;
    }
    return {
      labels: payload.labels,
      matrix: payload.matrix,
      mode: payload.mode ?? 'returns',
      notes: [`${payload.labels.length} assets`],
    };
  }, [optimizerResult]);

  const frontierPoints = useMemo(
    () =>
      frontierRows.map((point, idx) => ({
        idx,
        risk: point.risk,
        expectedReturn: point.expected_return,
        weights: point.weights,
      })),
    [frontierRows],
  );

  const sortedFrontierPoints = useMemo(
    () => [...frontierPoints].sort((a, b) => a.risk - b.risk),
    [frontierPoints],
  );

  const frontierAllocationSeries = useMemo(() => {
    if (!sortedFrontierPoints.length) return [];
    const assetSet = new Set<string>();
    sortedFrontierPoints.forEach((point) => {
      Object.keys(point.weights ?? {}).forEach((asset) => assetSet.add(asset));
    });
    return Array.from(assetSet).map((asset) => ({
      asset,
      values: sortedFrontierPoints.map((point) => ({
        risk: point.risk,
        weight: point.weights[asset] ?? 0,
        idx: point.idx,
      })),
    }));
  }, [sortedFrontierPoints]);

  const incumbentWeightMap = useMemo(() => {
    const map: Record<string, number> = {};
    optimizerWeights.forEach((row) => {
      map[row.asset] = row.weight;
    });
    return map;
  }, [optimizerWeights]);

  const incumbentLabelMap = useMemo(() => {
    const map: Record<string, string> = {};
    optimizerWeights.forEach((row) => {
      map[row.asset] = row.label;
    });
    return map;
  }, [optimizerWeights]);

  const selectedFrontierPoint =
    selectedFrontierIdx !== null && selectedFrontierIdx < frontierPoints.length
      ? frontierPoints[selectedFrontierIdx]
      : null;

  const frontierComparisonRows = useMemo(() => {
    if (!selectedFrontierPoint) return [];
    const assets = Array.from(
      new Set([
        ...Object.keys(selectedFrontierPoint.weights ?? {}),
        ...Object.keys(incumbentWeightMap),
      ]),
    ).sort();
    return assets.map((asset) => {
      const optimized = incumbentWeightMap[asset] ?? 0;
      const candidate = selectedFrontierPoint.weights[asset] ?? 0;
      return {
        asset,
        label: incumbentLabelMap[asset] ?? asset,
        incumbent: optimized,
        frontier: candidate,
        delta: candidate - optimized,
      };
    });
  }, [selectedFrontierPoint, incumbentWeightMap, incumbentLabelMap]);

  const frontierSignature = useMemo(
    () =>
      frontierRows
        .map(
          (point) =>
            `${point.expected_return.toFixed(6)}|${point.risk.toFixed(6)}|${Object.keys(point.weights).length}`,
        )
        .join(';'),
    [frontierRows],
  );

  const weightSignature = useMemo(() => {
    if (!optimizerWeights.length) return '';
    return optimizerWeights
      .map((row) => `${row.asset}:${row.weight.toFixed(6)}`)
      .sort()
      .join('|');
  }, [optimizerWeights]);

  const frontierStateSignature = useMemo(
    () => `${frontierSignature}|${weightSignature}`,
    [frontierSignature, weightSignature],
  );

  const handleFrontierSelect = useCallback((idx: number) => {
    setSelectedFrontierIdx(idx);
  }, []);

  useEffect(() => {
    if (!frontierPoints.length) {
      setSelectedFrontierIdx(null);
      lastFrontierSignature.current = null;
      return;
    }
    if (lastFrontierSignature.current === frontierStateSignature && selectedFrontierIdx !== null) {
      return;
    }
    if (!frontierStateSignature) return;
    lastFrontierSignature.current = frontierStateSignature;
    if (!Object.keys(incumbentWeightMap).length) {
      setSelectedFrontierIdx(0);
      return;
    }
    let closestIdx = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    frontierPoints.forEach((point, idx) => {
      let diff = 0;
      const assets = new Set([
        ...Object.keys(point.weights ?? {}),
        ...Object.keys(incumbentWeightMap),
      ]);
      assets.forEach((asset) => {
        diff += Math.abs((point.weights[asset] ?? 0) - (incumbentWeightMap[asset] ?? 0));
      });
      if (diff < bestDistance) {
        bestDistance = diff;
        closestIdx = idx;
      }
    });
    setSelectedFrontierIdx(closestIdx);
  }, [frontierPoints, frontierStateSignature, incumbentWeightMap, selectedFrontierIdx]);

  const selectedFrontierMetrics = selectedFrontierPoint
    ? {
        expected: selectedFrontierPoint.expectedReturn,
        risk: selectedFrontierPoint.risk,
      }
    : null;

  useEffect(() => {
    setRiskfolioContractEquity(accountEquity);
  }, [accountEquity]);

  useEffect(() => {

    try {

      const stored = typeof window !== 'undefined' ? localStorage.getItem(SELECTION_STORAGE_KEY) : null;

      if (stored) {

        const parsed = JSON.parse(stored) as { selection: Selection; includeDownsample?: boolean };

        if (parsed.selection) {

          setActiveSelection(parsed.selection);

        }

        if (typeof parsed.includeDownsample === 'boolean') {

          setIncludeDownsample(parsed.includeDownsample);

        }

      }

    } catch {

      // ignore corrupted storage

    }

  }, []);



  const persistedSelectionPayload = useMemo(
    () => ({ selection: activeSelection, includeDownsample }),
    [activeSelection, includeDownsample],
  );

  useEffect(() => {
    try {
      // Persist only the canonical selection + downsample flag; sandbox equity stays ephemeral.
      if (typeof window !== 'undefined') {
        localStorage.setItem(SELECTION_STORAGE_KEY, JSON.stringify(persistedSelectionPayload));
      }
    } catch {
      // ignore write failures
    }
  }, [persistedSelectionPayload]);



  useEffect(() => {
    if (apiMissing) return;
    (async () => {
      try {
        const meta = await getSelectionMeta();
        setSelectionMeta(meta);
        setFilesMeta(meta.files);
      } catch (error: any) {
        console.warn('Failed to load selection metadata', error);
        setErrorMessage(error?.message || 'Failed to load selection metadata');
      }
    })();
  }, [apiMissing]);

  useEffect(() => {
    if (activeSelection.accountEquity !== undefined && activeSelection.accountEquity !== null) {
      return;
    }
    const defaultEquity = selectionMeta?.account_equity ?? ACCOUNT_EQUITY_FALLBACK;
    setActiveSelection((prev) => {
      if (prev.accountEquity !== undefined && prev.accountEquity !== null) {
        return prev;
      }
      return {
        ...prev,
        accountEquity: defaultEquity,
      };
    });
  }, [selectionMeta, activeSelection.accountEquity]);


  const availableFiles = useMemo(() => (filesMeta.length ? filesMeta.map((f) => f.file_id).sort() : []), [filesMeta]);



  const fileLabelMap = useMemo(

    () => Object.fromEntries(filesMeta.map((f) => [f.file_id, f.filename])),

    [filesMeta],

  );



  const deriveFileMeta = useMemo(() => {
    return (fileId: string) => {

      const label = fileLabelMap[fileId] || fileId;

      const base = label.split('/').pop() || label;

      const parts = base.replace(/\.[^.]+$/, '').split('_');

      return {

        symbol: parts[1] || '',

        interval: parts[2] || '',

        strategy: parts.slice(3).join('_') || '',

      };

    };

  }, [fileLabelMap]);



  const fileMetaMap = useMemo(() => {
    const map = new Map<string, (typeof filesMeta)[number]>();
    filesMeta.forEach((file) => map.set(file.file_id, file));
    return map;
  }, [filesMeta]);

  const marginDefaultsByFile = useMemo(() => {
    if (!availableFiles.length) {
      return {};
    }
    const map: Record<string, number> = {};
    availableFiles.forEach((fileId) => {
      const meta = fileMetaMap.get(fileId);
      const symbol =
        (meta && meta.symbols.length
          ? meta.symbols[0]
          : deriveFileMeta(fileId).symbol || '').toUpperCase();
      const metaMargin = meta?.margin_per_contract;
      if (typeof metaMargin === 'number' && metaMargin > 0) {
        map[fileId] = metaMargin;
      } else if (symbol && MARGIN_DEFAULTS[symbol]) {
        map[fileId] = MARGIN_DEFAULTS[symbol];
      } else {
        map[fileId] = FALLBACK_MARGIN;
      }
    });
    return map;
  }, [availableFiles, fileMetaMap, deriveFileMeta]);


  const availableFilterValues = useMemo<FilterValueMap>(() => {
    if (!availableFiles.length) {
      return { symbols: [], intervals: [], strategies: [] };
    }

    const symbolSet = new Set<string>();
    const intervalSet = new Set<string>();
    const strategySet = new Set<string>();

    availableFiles.forEach((fileId) => {
      const meta = fileMetaMap.get(fileId);
      const fallback = deriveFileMeta(fileId);
      const symbols =
        meta && meta.symbols.length
          ? meta.symbols
          : fallback.symbol
            ? [fallback.symbol]
            : [];
      const intervals =
        meta && meta.intervals.length
          ? meta.intervals
          : fallback.interval
            ? [fallback.interval]
            : [];
      const strategies =
        meta && meta.strategies.length
          ? meta.strategies
          : fallback.strategy
            ? [fallback.strategy]
            : [];

      symbols.forEach((symbol) => symbolSet.add(symbol));
      intervals.forEach((interval) => intervalSet.add(String(interval)));
      strategies.forEach((strategy) => strategySet.add(strategy));
    });

    return {
      symbols: Array.from(symbolSet).sort(),
      intervals: Array.from(intervalSet).sort((a, b) => Number(a) - Number(b)),
      strategies: Array.from(strategySet).sort(),
    };
  }, [availableFiles, fileMetaMap, deriveFileMeta]);



  // Keep selection defaults aligned with ingest metadata (all files + all filters active).

  useEffect(() => {
    if (!availableFiles.length) return;
    setActiveSelection((prev) => {
      const labels =

        filesMeta.length > 0 ? Object.fromEntries(filesMeta.map((f) => [f.file_id, f.filename])) : prev.fileLabels ?? {};

      const prevLabels = prev.fileLabels ?? {};

      const labelKeys = Object.keys(labels);

      const labelChanged =

        filesMeta.length > 0 &&

        (Object.keys(prevLabels).length !== labelKeys.length || labelKeys.some((key) => prevLabels[key] !== labels[key]));



      const nextFiles = availableFiles;

      const nextSymbols = availableFilterValues.symbols;

      const nextIntervals = availableFilterValues.intervals;

      const nextStrategies = availableFilterValues.strategies;

      const nextDirection = 'All';

      const nextSpike = true;



      const shouldUpdate =

        labelChanged ||

        !arraysEqual(prev.files ?? [], nextFiles) ||

        !arraysEqual(prev.symbols ?? [], nextSymbols) ||

        !arraysEqual(prev.intervals ?? [], nextIntervals) ||

        !arraysEqual(prev.strategies ?? [], nextStrategies) ||

        (prev.direction ?? 'All') !== nextDirection ||

        prev.spike !== nextSpike;



      if (!shouldUpdate) return prev;



      return {

        ...prev,

        fileLabels: labels,

        files: nextFiles,

        symbols: nextSymbols,

        intervals: nextIntervals,

        strategies: nextStrategies,

        direction: nextDirection,

        spike: nextSpike,

      };

    });

  }, [availableFiles, availableFilterValues, filesMeta]);



  useEffect(() => {

    if (!availableFiles.length) return;

    setActiveSelection((prev) => {

      const existing = prev.contractMultipliers ?? prev.contracts ?? {};

      const allowed = new Set(availableFiles);

      let changed = !prev.contractMultipliers;

      const next: Record<string, number> = {};

      availableFiles.forEach((fileId) => {

        if (Object.prototype.hasOwnProperty.call(existing, fileId)) {

          next[fileId] = existing[fileId];

        } else {

          next[fileId] = 1;

          changed = true;

        }

      });

      Object.keys(existing).forEach((key) => {

        if (!allowed.has(key)) {

          changed = true;

        }

      });

      if (!changed) {

        return prev;

      }

      return {

        ...prev,

        contracts: next,

        contractMultipliers: next,

      };

    });

  }, [availableFiles]);

  useEffect(() => {
    if (!availableFiles.length) return;
    setActiveSelection((prev) => {
      const existing = prev.marginOverrides ?? prev.margins ?? {};
      const allowed = new Set(availableFiles);
      let changed = !prev.marginOverrides;
      const next: Record<string, number> = {};
      availableFiles.forEach((fileId) => {
        if (Object.prototype.hasOwnProperty.call(existing, fileId)) {
          next[fileId] = existing[fileId];
        } else {
          next[fileId] = marginDefaultsByFile[fileId] ?? FALLBACK_MARGIN;
          changed = true;
        }
      });
      Object.keys(existing).forEach((key) => {
        if (!allowed.has(key)) {
          changed = true;
        }
      });
      if (!changed) {
        return prev;
      }
      return {
        ...prev,
        margins: next,
        marginOverrides: next,
      };
    });
  }, [availableFiles, marginDefaultsByFile]);

  useEffect(() => {
    setMeanRiskOverrides((prev) => {
      if (!prev || Object.keys(prev).length === 0) {
        return prev;
      }
      const allowed = new Set(activeSelection.files);
      let changed = false;
      const next: BoundsOverrideState = {};
      Object.entries(prev).forEach(([asset, bounds]) => {
        if (allowed.has(asset)) {
          next[asset] = bounds;
        } else {
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [activeSelection.files]);

  useEffect(() => {
    if (!optimizerJobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const status = await fetchJobStatus(optimizerJobId);
        if (cancelled) return;
        setOptimizerStatus(status);
        if (status.status !== 'completed' && status.status !== 'failed') {
          timer = setTimeout(poll, 1500);
        }
      } catch (error) {
        if (!cancelled) {
          setOptimizerError(error instanceof Error ? error.message : 'Failed to fetch optimizer status');
        }
      }
    };

    poll();

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [optimizerJobId]);

  const matchesFilters = useCallback(
    (fileId: string) => {
      if (!availableFiles.length) {
        return false;
      }

      const normalizedSymbols = activeSelection.symbols.map((symbol) => symbol.toUpperCase());
      const normalizedIntervals = activeSelection.intervals.map((interval) => String(interval));
      const normalizedStrategies = activeSelection.strategies.map((strategy) => strategy.toUpperCase());
      const meta = fileMetaMap.get(fileId);
      const fallback = deriveFileMeta(fileId);
      const fileSymbols =
        meta && meta.symbols.length
          ? meta.symbols
          : fallback.symbol
            ? [fallback.symbol]
            : [];
      const fileIntervals =
        meta && meta.intervals.length
          ? meta.intervals
          : fallback.interval
            ? [fallback.interval]
            : [];
      const fileStrategies =
        meta && meta.strategies.length
          ? meta.strategies
          : fallback.strategy
            ? [fallback.strategy]
            : [];

      const requireSymbolFilters = availableFilterValues.symbols.length > 0;
      const requireIntervalFilters = availableFilterValues.intervals.length > 0;
      const requireStrategyFilters = availableFilterValues.strategies.length > 0;

      const symbolMatch =
        !requireSymbolFilters ||
        (normalizedSymbols.length > 0 &&
          fileSymbols.length > 0 &&
          fileSymbols.some((symbol) => normalizedSymbols.includes(symbol.toUpperCase())));
      const intervalMatch =
        !requireIntervalFilters ||
        (normalizedIntervals.length > 0 &&
          fileIntervals.length > 0 &&
          fileIntervals.some((interval) => normalizedIntervals.includes(String(interval))));
      const strategyMatch =
        !requireStrategyFilters ||
        (normalizedStrategies.length > 0 &&
          fileStrategies.length > 0 &&
          fileStrategies.some((strategy) => normalizedStrategies.includes(strategy.toUpperCase())));

      return symbolMatch && intervalMatch && strategyMatch;
    },
    [
      activeSelection.symbols,
      activeSelection.intervals,
      activeSelection.strategies,
      deriveFileMeta,
      fileMetaMap,
      availableFilterValues,
      availableFiles,
    ],
  );



  const filteredFileIds = useMemo(() => {

    if (!activeSelection.files || !activeSelection.files.length) return [];

    return activeSelection.files.filter((fileId) => matchesFilters(fileId));

  }, [activeSelection.files, matchesFilters]);

  const hasFiles = filteredFileIds.length > 0;
  const canQueryData = hasFiles && !apiMissing;



  const filteredFileSet = useMemo(() => new Set(filteredFileIds), [filteredFileIds]);



  const selectionForFetch = useMemo(
    () => normalizeSelection({ ...activeSelection, files: filteredFileIds, accountEquity }),
    [activeSelection, filteredFileIds, accountEquity],
  );


  const selectionKey = useMemo(() => JSON.stringify(selectionForFetch), [selectionForFetch]);

  const riskfolioContracts = useMemo<RiskfolioSuggestions>(() => {
    const equity = Math.max(0, coerceNumber(riskfolioContractEquity, 0));
    const note = `Account equity: ${formatCurrency(equity)}. Contracts = floor(weight * account equity / the margin input on Load Trade Lists).`;
    const allocations = optimizerWeights;
    if (!allocations.length) {
      return { rows: [], summary: null, note };
    }

    const contractMap = activeSelection.contractMultipliers ?? activeSelection.contracts ?? {};
    const marginMap = activeSelection.marginOverrides ?? activeSelection.margins ?? {};

    const rows: RiskfolioSuggestedRow[] = [];
    let totalSuggestedContracts = 0;
    let totalCurrentContracts = 0;
    let totalSuggestedMargin = 0;
    let totalCurrentMargin = 0;
    let maxSuggestedGap: number | null = null;
    let maxCurrentGap: number | null = null;
    let suggestedGapSum = 0;
    let currentGapSum = 0;
    let activeWeightCount = 0;

    allocations.forEach((allocation) => {
      const asset = allocation.asset;
      const fileLabel = fileLabelMap[asset] || allocation.label || asset;
      const weight = coerceNumber(allocation.weight, 0);
      const marginValue =
        (marginMap && Object.prototype.hasOwnProperty.call(marginMap, asset) ? marginMap[asset] : undefined) ??
        allocation.margin_per_contract;
      const marginPerContract = coerceNumber(marginValue, 0);
      const currentContractsRaw =
        (contractMap && Object.prototype.hasOwnProperty.call(contractMap, asset) ? contractMap[asset] : undefined) ?? 0;
      const currentContracts = Math.round(coerceNumber(currentContractsRaw, 0));
      const allocationNotional = weight * equity;
      const suggestedContracts =
        marginPerContract > 0 ? Math.max(0, Math.floor(allocationNotional / marginPerContract)) : 0;
      const suggestedMargin = suggestedContracts * marginPerContract;
      const currentMargin = currentContracts * marginPerContract;

      totalSuggestedContracts += Math.max(0, suggestedContracts);
      totalCurrentContracts += Math.max(0, currentContracts);
      totalSuggestedMargin += Math.max(0, suggestedMargin);
      totalCurrentMargin += Math.max(0, currentMargin);

      let suggestedGap: number | null = null;
      let currentGap: number | null = null;

      if (equity > 0 && Number.isFinite(suggestedMargin)) {
        const impliedWeight = suggestedMargin / equity;
        suggestedGap = weight - impliedWeight;
        const gapAbs = Math.abs(suggestedGap);
        maxSuggestedGap = maxSuggestedGap === null ? gapAbs : Math.max(maxSuggestedGap, gapAbs);
        suggestedGapSum += gapAbs;
      }

      if (equity > 0 && Number.isFinite(currentMargin)) {
        const impliedWeight = currentMargin / equity;
        currentGap = weight - impliedWeight;
        const gapAbs = Math.abs(currentGap);
        maxCurrentGap = maxCurrentGap === null ? gapAbs : Math.max(maxCurrentGap, gapAbs);
        currentGapSum += gapAbs;
      }

      if (weight !== 0) {
        activeWeightCount += 1;
      }

      rows.push({
        asset,
        label: fileLabel,
        weight,
        marginPerContract,
        suggestedContracts,
        currentContracts,
        suggestedMargin,
        currentMargin,
        suggestedGap,
        currentGap,
        delta: suggestedContracts - currentContracts,
      });
    });

    const summary: RiskfolioSuggestedSummary | null = rows.length
      ? {
          suggestedContractsTotal: totalSuggestedContracts,
          currentContractsTotal: totalCurrentContracts,
          suggestedMarginTotal: totalSuggestedMargin,
          currentMarginTotal: totalCurrentMargin,
          maxSuggestedGap,
          avgSuggestedGap: activeWeightCount ? suggestedGapSum / activeWeightCount : null,
          maxCurrentGap,
          avgCurrentGap: activeWeightCount ? currentGapSum / activeWeightCount : null,
        }
      : null;

    return { rows, summary, note };
  }, [
    optimizerResult,
    activeSelection.contractMultipliers,
    activeSelection.contracts,
    activeSelection.marginOverrides,
    activeSelection.margins,
    riskfolioContractEquity,
    fileLabelMap,
  ]);
  useEffect(() => {
    setRiskfolioApplyMessage(null);
  }, [riskfolioContracts]);

  useEffect(() => {
    setOptimizerStatus(null);
    setOptimizerJobId(null);
    setOptimizerError(null);
    setOptimizerLoading(false);
  }, [selectionKey]);
  useEffect(() => {
    setMetricsRequested(false);
  }, [selectionKey]);

  const handleOverrideChange = useCallback((fileId: string, field: 'min' | 'max', value: string) => {
    const sanitizedValue = sanitizeWeightInputString(value);
    setMeanRiskOverrides((prev) => {
      const next = { ...prev };
      const existing = next[fileId] ?? { min: '', max: '' };
      const updated = { ...existing, [field]: sanitizedValue };
      const hasValue =
        (updated.min && updated.min.trim() !== '') || (updated.max && updated.max.trim() !== '');
      if (!hasValue) {
        if (next[fileId]) {
          delete next[fileId];
          return { ...next };
        }
        return prev;
      }
      next[fileId] = updated;
      return next;
    });
  }, []);

  const handleApplyRiskfolioContracts = useCallback(() => {
    if (!riskfolioContracts.rows.length) {
      setRiskfolioApplyMessage('Run the optimizer to generate suggested contracts.');
      return;
    }
    setActiveSelection((prev) => {
      const current = prev.contractMultipliers ?? prev.contracts ?? {};
      const next = { ...current };
      let changed = false;
      riskfolioContracts.rows.forEach((row) => {
        if (next[row.asset] !== row.suggestedContracts) {
          next[row.asset] = row.suggestedContracts;
          changed = true;
        }
      });
      if (!changed) {
        return prev;
      }
      return {
        ...prev,
        contracts: next,
        contractMultipliers: next,
      };
    });
    setRiskfolioApplyMessage(
      `Applied Riskfolio suggested contracts to ${riskfolioContracts.rows.length} file${riskfolioContracts.rows.length === 1 ? '' : 's'}.`,
    );
  }, [riskfolioContracts, setActiveSelection]);

  const formatRiskfolioGap = (gap: number | null) => (gap === null ? 'n/a' : formatPercent(gap));
  const formatRiskfolioGapSummary = (max: number | null, avg: number | null) => {
    if (max === null) return 'Max: n/a';
    const avgText = avg === null ? 'n/a' : formatPercent(avg);
    return `Max: ${formatPercent(max)} / Avg: ${avgText}`;
  };
  const canApplyRiskfolio = riskfolioContracts.rows.length > 0;

  const handleOptimize = useCallback(async () => {
    if (!filteredFileIds.length) {
      setOptimizerError('Select at least one file before optimizing.');
      return;
    }
    setOptimizerError(null);
    setOptimizerStatus(null);
    setOptimizerJobId(null);
    setOptimizerLoading(true);
    try {
      const boundsOverrides = Object.entries(meanRiskOverrides).reduce(
        (acc, [asset, bounds]) => {
          const minVal = parseOptionalWeightInputValue(bounds.min ?? null);
          const maxVal = parseOptionalWeightInputValue(bounds.max ?? null);
          if (minVal === null && maxVal === null) {
            return acc;
          }
          acc[asset] = [minVal, maxVal];
          return acc;
        },
        {} as Record<string, [number | null, number | null]>,
      );
      const payload: MeanRiskPayload = {
        objective: meanRiskObjective,
        risk_measure: meanRiskRiskMeasure,
        return_model: meanRiskReturnModel,
        method_mu: meanRiskReturnEstimate,
        method_cov: meanRiskCovariance,
        solver: null,
        risk_free_rate: meanRiskRiskFree / 100,
        risk_aversion: meanRiskRiskAversion,
        alpha: meanRiskAlpha,
        budget: meanRiskBudget,
        bounds: {
          default_min: minWeightBound,
          default_max: maxWeightBound,
          overrides: boundsOverrides,
        },
        symbol_caps: parseCapsInput(meanRiskSymbolCaps),
        strategy_caps: parseCapsInput(meanRiskStrategyCaps),
        efficient_frontier_points: meanRiskFrontierPoints,
        max_risk: parseOptionalNumber(meanRiskMaxRisk),
        min_return: parseOptionalNumber(meanRiskMinReturn),
        turnover_limit: parseOptionalNumber(meanRiskTurnover),
      };
      const response = await submitRiskfolioJob(selectionForFetch, payload);
      setOptimizerJobId(response.job_id);
    } catch (error) {
      setOptimizerError(error instanceof Error ? error.message : 'Failed to run optimizer');
    } finally {
      setOptimizerLoading(false);
    }
  }, [
    filteredFileIds.length,
    selectionForFetch,
    meanRiskObjective,
    meanRiskRiskMeasure,
    meanRiskReturnModel,
    meanRiskReturnEstimate,
    meanRiskCovariance,
    meanRiskRiskFree,
    meanRiskRiskAversion,
    meanRiskAlpha,
    meanRiskBudget,
    minWeightBound,
    maxWeightBound,
    meanRiskSymbolCaps,
    meanRiskStrategyCaps,
    meanRiskMaxRisk,
    meanRiskMinReturn,
    meanRiskTurnover,
    meanRiskFrontierPoints,
    meanRiskOverrides,
  ]);



  const equityQuery = useQuery({

    queryKey: ['equity', selectionKey, includeDownsample],

    queryFn: () => fetchSeries(selectionForFetch, 'equity', { downsample: includeDownsample }),

    staleTime: STALE_TIME,

    enabled: canQueryData,

  });

  const equityPctQuery = useQuery({

    queryKey: ['equityPercent', selectionKey, includeDownsample],

    queryFn: () => fetchSeries(selectionForFetch, 'equityPercent', { downsample: includeDownsample }),

    staleTime: STALE_TIME,

    enabled: canQueryData,

  });

  const drawdownQuery = useQuery({

    queryKey: ['drawdown', selectionKey, includeDownsample],

    queryFn: () => fetchSeries(selectionForFetch, 'drawdown', { downsample: includeDownsample }),

    staleTime: STALE_TIME,

    enabled: canQueryData,

  });

  const intradayDdQuery = useQuery({

    queryKey: ['intradayDrawdown', selectionKey, includeDownsample],

    queryFn: () => fetchSeries(selectionForFetch, 'intradayDrawdown', { downsample: includeDownsample }),

    staleTime: STALE_TIME,

    enabled: canQueryData,

  });

  const netposQuery = useQuery({

    queryKey: ['netpos', selectionKey, includeDownsample],

    queryFn: () => fetchSeries(selectionForFetch, 'netpos', { downsample: includeDownsample }),

    staleTime: STALE_TIME,

    enabled: canQueryData,

  });

  const marginQuery = useQuery({

    queryKey: ['margin', selectionKey, includeDownsample],

    queryFn: () => fetchSeries(selectionForFetch, 'margin', { downsample: includeDownsample }),

    staleTime: STALE_TIME,

    enabled: canQueryData,

  });

  const netposPerFileQuery = useQuery({
    queryKey: ['netpos', selectionKey, includeDownsample, 'per_file'],
    queryFn: () =>
      fetchSeries(selectionForFetch, 'netpos', { downsample: includeDownsample, exposureView: 'per_file' }),
    staleTime: STALE_TIME,
    enabled: canQueryData && showExposureDebug,
  });

  const netposPerSymbolQuery = useQuery({
    queryKey: ['netpos', selectionKey, includeDownsample, 'per_symbol'],
    queryFn: () =>
      fetchSeries(selectionForFetch, 'netpos', { downsample: includeDownsample, exposureView: 'per_symbol' }),
    staleTime: STALE_TIME,
    enabled: canQueryData && showExposureDebug,
  });

  const netposPortfolioStepQuery = useQuery({
    queryKey: ['netpos', selectionKey, includeDownsample, 'portfolio_step'],
    queryFn: () =>
      fetchSeries(selectionForFetch, 'netpos', { downsample: includeDownsample, exposureView: 'portfolio_step' }),
    staleTime: STALE_TIME,
    enabled: canQueryData && showExposureDebug,
  });

  const marginPerFileQuery = useQuery({
    queryKey: ['margin', selectionKey, includeDownsample, 'per_file'],
    queryFn: () =>
      fetchSeries(selectionForFetch, 'margin', { downsample: includeDownsample, exposureView: 'per_file' }),
    staleTime: STALE_TIME,
    enabled: canQueryData && showExposureDebug,
  });

  const marginPerSymbolQuery = useQuery({
    queryKey: ['margin', selectionKey, includeDownsample, 'per_symbol'],
    queryFn: () =>
      fetchSeries(selectionForFetch, 'margin', { downsample: includeDownsample, exposureView: 'per_symbol' }),
    staleTime: STALE_TIME,
    enabled: canQueryData && showExposureDebug,
  });

  const marginPortfolioStepQuery = useQuery({
    queryKey: ['margin', selectionKey, includeDownsample, 'portfolio_step'],
    queryFn: () =>
      fetchSeries(selectionForFetch, 'margin', { downsample: includeDownsample, exposureView: 'portfolio_step' }),
    staleTime: STALE_TIME,
    enabled: canQueryData,
  });

  const histogramQuery = useQuery({

    queryKey: ['histogram', selectionKey],

    queryFn: () => fetchHistogram(selectionForFetch),

    staleTime: STALE_TIME,

    enabled: canQueryData,

  });

  const metricsQuery = useQuery({

    queryKey: ['metrics', selectionKey, metricsRequested],

    queryFn: () => fetchMetrics(selectionForFetch),

    staleTime: STALE_TIME,

    enabled: canQueryData && metricsRequested,

  });



  useEffect(() => {

    const points = equityQuery.data?.portfolio ?? [];

    const timestamps = points.map((p) => p.timestamp).filter(Boolean) ?? [];

    if (!timestamps.length) return;

    const sorted = [...timestamps].sort();

    const toDate = (ts: string) => ts.slice(0, 10);

    const rangeStart = toDate(sorted[0]);

    const rangeEnd = toDate(sorted[sorted.length - 1]);

    setActiveSelection((prev) => {

      const nextStart = prev.start ?? rangeStart;

      const nextEnd = prev.end ?? rangeEnd;

      if (nextStart === prev.start && nextEnd === prev.end) return prev;

      return { ...prev, start: nextStart, end: nextEnd };

    });

  }, [equityQuery.data, activeSelection.name]);



  const equityLines = useMemo(() => {

    const portfolioPoints = (equityQuery.data?.portfolio ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));

    const perFile = (equityQuery.data?.perFile ?? []).map((line) => ({

      name: line.label || line.contributor_id,

      points: line.points.map((p) => ({ timestamp: p.timestamp, value: p.value })),

    }));

    return { perFile, portfolio: portfolioPoints };

  }, [equityQuery.data]);



  const equityPercentLines = useMemo(() => {

    const portfolioPoints = (equityPctQuery.data?.portfolio ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));

    const perFile = (equityPctQuery.data?.perFile ?? []).map((line) => ({

      name: line.label || line.contributor_id,

      points: line.points.map((p) => ({ timestamp: p.timestamp, value: p.value })),

    }));

    return { perFile, portfolio: portfolioPoints };

  }, [equityPctQuery.data]);



  useEffect(() => {

    const names = new Set<string>();

    equityLines.perFile.forEach((s) => names.add(s.name));

    equityPercentLines.perFile.forEach((s) => names.add(s.name));

    names.add('Portfolio');

    setPlotEnabled((prev) => {

      const next = { ...prev };

      names.forEach((name) => {

        if (next[name] === undefined) next[name] = true;

      });

      Object.keys(next).forEach((key) => {

        if (!names.has(key)) delete next[key];

      });

      return next;

    });

  }, [equityLines.perFile, equityPercentLines.perFile]);



  const drawdownLines = useMemo(() => {

    const portfolioPoints = (drawdownQuery.data?.portfolio ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));

    const perFile = (drawdownQuery.data?.perFile ?? []).map((line) => ({

      name: line.label || line.contributor_id,

      points: line.points.map((p) => ({ timestamp: p.timestamp, value: p.value })),

    }));

    return { perFile, portfolio: portfolioPoints };

  }, [drawdownQuery.data]);



  const drawdownPercentLines = useMemo(() => {

    const equityBase = accountEquity || 1;

    return {

      perFile: drawdownLines.perFile.map((s) => ({

        name: s.name,

        points: s.points.map((p) => ({ ...p, value: (p.value / equityBase) * 100 })),

      })),

      portfolio: drawdownLines.portfolio.map((p) => ({ ...p, value: (p.value / equityBase) * 100 })),

    };

  }, [drawdownLines, accountEquity]);



  useEffect(() => {

    const names = new Set<string>();

    drawdownLines.perFile.forEach((s) => names.add(s.name));

    names.add('Portfolio');

    setPlotDrawdownEnabled((prev) => {

      const next = { ...prev };

      names.forEach((name) => {

        if (next[name] === undefined) next[name] = true;

      });

      Object.keys(next).forEach((key) => {

        if (!names.has(key)) delete next[key];

      });

      return next;

    });

  }, [drawdownLines.perFile]);



  const buildSeriesLines = useCallback((series?: SeriesResponse) => {
    const portfolioPoints = (series?.portfolio ?? []).map((p) => ({ timestamp: p.timestamp, value: p.value }));
    const perFile = (series?.perFile ?? []).map((line) => ({
      name: line.label || line.contributor_id,
      points: line.points.map((p) => ({ timestamp: p.timestamp, value: p.value })),
    }));
    return { perFile, portfolio: portfolioPoints };
  }, []);

  const buildStepper = useCallback((points: { timestamp: string; value: number }[], fallback: number) => {
    if (!points.length) {
      return () => fallback;
    }
    const sorted = [...points].sort((a, b) => {
      if (a.timestamp === b.timestamp) return 0;
      return a.timestamp < b.timestamp ? -1 : 1;
    });
    let idx = 0;
    let last = fallback;
    return (ts: string) => {
      while (idx < sorted.length && sorted[idx].timestamp <= ts) {
        last = sorted[idx].value;
        idx += 1;
      }
      return last;
    };
  }, []);

  const marginLines = useMemo(() => buildSeriesLines(marginQuery.data), [marginQuery.data, buildSeriesLines]);



  const netposLines = useMemo(() => buildSeriesLines(netposQuery.data), [netposQuery.data, buildSeriesLines]);

  const marginPerFileLines = useMemo(() => buildSeriesLines(marginPerFileQuery.data), [
    marginPerFileQuery.data,
    buildSeriesLines,
  ]);
  const marginPerSymbolLines = useMemo(() => buildSeriesLines(marginPerSymbolQuery.data), [
    marginPerSymbolQuery.data,
    buildSeriesLines,
  ]);
  const marginPortfolioStepLines = useMemo(() => buildSeriesLines(marginPortfolioStepQuery.data), [
    marginPortfolioStepQuery.data,
    buildSeriesLines,
  ]);
  const netposPerFileLines = useMemo(() => buildSeriesLines(netposPerFileQuery.data), [
    netposPerFileQuery.data,
    buildSeriesLines,
  ]);
  const netposPerSymbolLines = useMemo(() => buildSeriesLines(netposPerSymbolQuery.data), [
    netposPerSymbolQuery.data,
    buildSeriesLines,
  ]);
  const netposPortfolioStepLines = useMemo(() => buildSeriesLines(netposPortfolioStepQuery.data), [
    netposPortfolioStepQuery.data,
    buildSeriesLines,
  ]);



  const purchasingPowerLines = useMemo(() => {

    const buildTimeline = (left: { timestamp: string }[], right: { timestamp: string }[]) =>
      Array.from(
        new Set([
          ...left.map((p) => p.timestamp),
          ...right.map((p) => p.timestamp),
        ]),
      ).sort();

    const buildSeries = (
      equityPts: { timestamp: string; value: number }[],
      marginPts: { timestamp: string; value: number }[],
    ) => {
      const timeline = buildTimeline(equityPts, marginPts);
      const equityAt = buildStepper(equityPts, accountEquity);
      const marginAt = buildStepper(marginPts, 0);
      return timeline.map((ts) => {
        const equityVal = equityAt(ts);
        const marginVal = marginAt(ts);
        return { timestamp: ts, value: equityVal - marginVal };
      });
    };



    const perFile = equityLines.perFile.map((eq) => {

      const margin = marginLines.perFile.find((m) => m.name === eq.name);

      const marginPoints = margin?.points ?? [];

      return { name: eq.name, points: buildSeries(eq.points, marginPoints) };

    });



    const portfolioPoints = buildSeries(equityLines.portfolio, marginLines.portfolio);



    return { perFile, portfolio: portfolioPoints };

  }, [equityLines, marginLines, accountEquity, buildStepper]);

  const startDateSafetyLine = useMemo(() => {
    const equityPts = equityLines.portfolio;
    if (!equityPts.length) return [];
    const sortedEquity = [...equityPts].sort((a, b) => (a.timestamp < b.timestamp ? -1 : 1));
    const marginAt = buildStepper(marginPortfolioStepLines.portfolio, 0);

    const aValues = sortedEquity.map((p) => ({
      timestamp: p.timestamp,
      equity: p.value,
      a: accountEquity + p.value - marginAt(p.timestamp),
    }));

    const suffixMin: number[] = new Array(aValues.length);
    let minSoFar = Number.POSITIVE_INFINITY;
    for (let i = aValues.length - 1; i >= 0; i -= 1) {
      minSoFar = Math.min(minSoFar, aValues[i].a);
      suffixMin[i] = minSoFar;
    }

    return aValues.map((row, idx) => ({
      timestamp: row.timestamp,
      value: suffixMin[idx] - row.equity,
    }));
  }, [equityLines.portfolio, marginPortfolioStepLines.portfolio, accountEquity, buildStepper]);



  const purchasingPowerDrawdownLines = useMemo(() => {

    const toDrawdown = (points: { timestamp: string; value: number }[]) => {

      let maxSoFar = Number.NEGATIVE_INFINITY;

      return points.map((p) => {

        maxSoFar = Math.max(maxSoFar, p.value);

        return { ...p, value: p.value - maxSoFar };

      });

    };

    return {

      perFile: purchasingPowerLines.perFile.map((s) => ({ name: s.name, points: toDrawdown(s.points) })),

      portfolio: toDrawdown(purchasingPowerLines.portfolio),

    };

  }, [purchasingPowerLines]);



  const purchasingPowerDrawdownReferenceLine = useMemo(() => {

    const base = accountEquity ? accountEquity * -1 : 0;

    if (!base || !purchasingPowerDrawdownLines.portfolio.length) return [];

    return purchasingPowerDrawdownLines.portfolio.map((point) => ({

      timestamp: point.timestamp,

      value: base,

    }));

  }, [accountEquity, purchasingPowerDrawdownLines.portfolio]);



  useEffect(() => {

    const names = new Set<string>();

    purchasingPowerLines.perFile.forEach((s) => names.add(s.name));

    names.add('Portfolio');

    setPlotMarginEnabled((prev) => {

      const next = { ...prev };

      names.forEach((name) => {

        if (next[name] === undefined) next[name] = true;

      });

      Object.keys(next).forEach((key) => {

        if (!names.has(key)) delete next[key];

      });

      return next;

    });

  }, [purchasingPowerLines.perFile]);



  const histogramData = useMemo(() => {

    const portfolioBuckets = histogramQuery.data?.buckets ?? [];

    return portfolioBuckets.length ? [{ name: 'Portfolio', buckets: portfolioBuckets }] : [];

  }, [histogramQuery.data]);



  useEffect(() => {

    const names = new Set<string>();

    histogramData.forEach((h) => names.add(h.name));

    setPlotHistogramEnabled((prev) => {

      const next = { ...prev };

      names.forEach((name) => {

        if (next[name] === undefined) next[name] = true;

      });

      Object.keys(next).forEach((key) => {

        if (!names.has(key)) delete next[key];

      });

      return next;

    });

  }, [histogramData]);





  const metricsSummary = useMemo(() => {

    if (!metricsRequested) return null;

    const portfolioMetrics = metricsQuery.data?.portfolio?.metrics;

    if (!portfolioMetrics) return null;

    const netProfit = coerceNumber(portfolioMetrics.total_net_profit, 0);

    const drawdown = coerceNumber(portfolioMetrics.close_to_close_drawdown_value, 0);

    const totalTrades = coerceNumber(portfolioMetrics.total_trades, 0);

    const winRate = coerceNumber(portfolioMetrics.percent_profitable, 0);

    return { netProfit, drawdown, totalTrades, winRate };

  }, [metricsQuery.data, metricsRequested]);
  const metricsBlocks = useMemo(() => {
    if (!metricsRequested || !metricsQuery.data) return [];
    const { portfolio, files } = metricsQuery.data;
    const ordered = [];
    if (portfolio) ordered.push(portfolio);
    if (files?.length) ordered.push(...files);
    return ordered;
  }, [metricsQuery.data, metricsRequested]);



  const exposureDebugFetching =
    showExposureDebug &&
    (netposPerFileQuery.isFetching ||
      netposPerSymbolQuery.isFetching ||
      netposPortfolioStepQuery.isFetching ||
      marginPerFileQuery.isFetching ||
      marginPerSymbolQuery.isFetching);

  const busy =

    equityQuery.isFetching ||

    drawdownQuery.isFetching ||

    metricsQuery.isFetching ||

    equityPctQuery.isFetching ||

    intradayDdQuery.isFetching ||

    netposQuery.isFetching ||

    marginQuery.isFetching ||

    histogramQuery.isFetching ||

    marginPortfolioStepQuery.isFetching ||
    exposureDebugFetching;



  const activeBadge = busy ? <div className="badge">Loading...</div> : <div className="badge">Live</div>;

  const renderUploadPlaceholder = (label: string) => (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="placeholder-text">Upload files to view {label}.</div>

    </div>

  );
  const handleMetricsComputation = () => {

    if (!canQueryData) return;

    setMetricsRequested(true);

    void metricsQuery.refetch();

  };



  const renderSummary = () => {

    if (!hasFiles) {
      return renderUploadPlaceholder('the summary tab');
    }

    return (

    <div>

      <div className="flex gap-md" style={{ alignItems: 'center', justifyContent: 'space-between' }}>

        <h2 className="section-title" style={{ margin: 0 }}>

          Equity, drawdown, and metrics

        </h2>

        {activeBadge}

      </div>



      <div className="metric-cards" style={{ marginTop: 14 }}>

        <div className="metric-card">

          <span className="text-muted small">Net Profit (portfolio)</span>

          <strong>{metricsSummary ? formatCurrency(metricsSummary.netProfit) : '--'}</strong>

        </div>

        <div className="metric-card">

          <span className="text-muted small">Close-to-close Drawdown</span>

          <strong>{metricsSummary ? formatCurrency(metricsSummary.drawdown) : '--'}</strong>

        </div>

        <div className="metric-card">

          <span className="text-muted small">Total Trades</span>

          <strong>{metricsSummary ? metricsSummary.totalTrades.toLocaleString() : '--'}</strong>

        </div>

        <div className="metric-card">

          <span className="text-muted small">Win %</span>

          <strong>{metricsSummary ? `${metricsSummary.winRate.toFixed(1)}%` : '--'}</strong>

        </div>

        <div className="metric-card">

          <span className="text-muted small">Files</span>

          <strong>{filteredFileIds.length}</strong>

        </div>

        <div className="metric-card">

          <span className="text-muted small">Account Equity</span>

          <strong>{`$${accountEquity.toLocaleString()}`}</strong>

        </div>

      </div>



      <div className="charts-grid" style={{ marginTop: 18 }}>

        <div className="card">

          {equityQuery.isError ? (

            <div className="placeholder-text">{getQueryErrorMessage(equityQuery.error)}</div>

          ) : equityQuery.data ? (

            <SeriesChart title="Equity Curve" series={equityQuery.data} color="#4cc3ff" />

          ) : (

            <div className="placeholder-text">No equity data.</div>

          )}

          <div className="text-muted small">

            Points: {equityQuery.data?.downsampled_count ?? equityQuery.data?.portfolio.length ?? 0}

          </div>

        </div>

        <div className="card">

          {equityPctQuery.isError ? (

            <div className="placeholder-text">{getQueryErrorMessage(equityPctQuery.error)}</div>

          ) : equityPctQuery.data ? (

            <SeriesChart title="Percent Equity" series={equityPctQuery.data} color="#8fe3c7" />

          ) : (

            <div className="placeholder-text">No percent equity data.</div>

          )}

          <div className="text-muted small">

            Points: {equityPctQuery.data?.downsampled_count ?? equityPctQuery.data?.portfolio.length ?? 0}

          </div>

        </div>

        <div className="card">

          {drawdownQuery.isError ? (

            <div className="placeholder-text">{getQueryErrorMessage(drawdownQuery.error)}</div>

          ) : drawdownQuery.data ? (

            <SeriesChart title="Drawdown" series={drawdownQuery.data} color="#ff8f6b" />

          ) : (

            <div className="placeholder-text">No drawdown data.</div>

          )}

          <div className="text-muted small">

            Points: {drawdownQuery.data?.downsampled_count ?? drawdownQuery.data?.portfolio.length ?? 0}

          </div>

        </div>

        <div className="card">

          {intradayDdQuery.isError ? (

            <div className="placeholder-text">{getQueryErrorMessage(intradayDdQuery.error)}</div>

          ) : intradayDdQuery.data ? (

            <SeriesChart title="Intraday Drawdown" series={intradayDdQuery.data} color="#f4c95d" />

          ) : (

            <div className="placeholder-text">No intraday drawdown data.</div>

          )}

          <div className="text-muted small">

            Points: {intradayDdQuery.data?.downsampled_count ?? intradayDdQuery.data?.portfolio.length ?? 0}

          </div>

        </div>

        <div className="card">

          {netposQuery.isError ? (

            <div className="placeholder-text">{getQueryErrorMessage(netposQuery.error)}</div>

          ) : netposQuery.data ? (

            <SeriesChart title="Net Position" series={netposQuery.data} color="#9f8bff" />

          ) : (

            <div className="placeholder-text">No net position data.</div>

          )}

          <div className="text-muted small">

            Points: {netposQuery.data?.downsampled_count ?? netposQuery.data?.portfolio.length ?? 0}

          </div>

        </div>

        <div className="card">

          {marginQuery.isError ? (

            <div className="placeholder-text">{getQueryErrorMessage(marginQuery.error)}</div>

          ) : marginQuery.data ? (

            <SeriesChart title="Margin Usage" series={marginQuery.data} color="#54ffd0" />

          ) : (

            <div className="placeholder-text">No margin data.</div>

          )}

          <div className="text-muted small">

            Points: {marginQuery.data?.downsampled_count ?? marginQuery.data?.portfolio.length ?? 0}

          </div>

        </div>

        <div className="card">

          {histogramQuery.isError ? (

            <div className="placeholder-text">{getQueryErrorMessage(histogramQuery.error)}</div>

          ) : histogramQuery.data ? (

            <HistogramChart histogram={histogramQuery.data} />

          ) : (

            <div className="placeholder-text">No histogram data.</div>

          )}

          <div className="text-muted small">

            Distribution: {histogramQuery.data?.buckets.length ?? 0} buckets

          </div>

        </div>

      </div>



      <div style={{ marginTop: 18 }}>

        <h3 className="section-title">Per-file metrics</h3>

        {!metricsRequested ? (

          <div className="placeholder-text">Compute metrics from the Metrics tab to populate this table.</div>

        ) : metricsQuery.isError ? (

          <div className="placeholder-text">{getQueryErrorMessage(metricsQuery.error)}</div>

        ) : metricsBlocks.length ? (

          <MetricsGrid blocks={metricsBlocks} />

        ) : metricsQuery.isFetching ? (

          <div className="placeholder-text">Computing metrics...</div>

        ) : (

          <div className="placeholder-text">No metrics returned yet.</div>

        )}

      </div>

    </div>

    );

  };



  const renderEquityCurves = () => {

    if (!hasFiles) {
      return renderUploadPlaceholder('equity curves');
    }

    return (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Equity Curves</h3>

        {activeBadge}

      </div>

      <div className="text-muted small" style={{ marginTop: 6 }}>

        Equity curves honor filters (symbols/intervals/strategies), contract multipliers, and date range. Setting contracts to zero or toggling off filters will exclude that file from the chart and portfolio line.

      </div>

      <div className="card" style={{ marginTop: 12 }}>

        <strong>Plot lines</strong>

        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>

          {[...equityLines.perFile.map((s) => s.name), 'Portfolio'].map((name) => {

            const active = plotEnabled[name] ?? true;

            return (

              <button

                key={name}

                type="button"

                className={`chip ${active ? 'chip-active' : ''}`}

                onClick={() => setPlotEnabled((prev) => ({ ...prev, [name]: !active }))}

              >

                {name}

              </button>

            );

          })}

        </div>

      </div>



      <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>

        <EquityMultiChart

          title="Equity Curve ($)"

          series={[

            ...equityLines.perFile.filter((s) => plotEnabled[s.name] !== false),

            ...(plotEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: equityLines.portfolio }]),

          ]}

        />

        <EquityMultiChart

          title="Equity Curve (%)"

          series={[

            ...equityPercentLines.perFile.filter((s) => plotEnabled[s.name] !== false),

            ...(plotEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: equityPercentLines.portfolio }]),

          ]}

        />

      </div>

    </div>

    );

  };



  const renderCorrelations = () => (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Correlations</h3>

        <div className="badge">Coming soon</div>

      </div>

      <p className="text-muted small" style={{ marginTop: 4 }}>

        Correlation heatmaps will return once the backend routes are wired to real data. Upload files now so they’re ready
        to use when the feature ships.

      </p>

      <div className="card" style={{ marginTop: 14 }}>

        <div className="placeholder-text">

          {hasFiles

            ? 'Correlation matrix calculations are coming soon.'

            : 'Upload files to view correlations once this module is available.'}

        </div>

      </div>

    </div>

  );



  const renderOptimizer = () => {
    if (!hasFiles) {
      return renderUploadPlaceholder('the Riskfolio optimizer');
    }
    const summary = optimizerResult?.summary;
  const contractRows = optimizerResult?.contracts ?? [];
  const weightRows = optimizerWeights;
  const statusLabel = optimizerStatus?.status ?? (optimizerJobId ? 'running' : 'idle');
  const progressWidth = Math.min(
    100,
    optimizerRunning || optimizerLoading ? Math.max(optimizerProgress, 10) : optimizerProgress,
  );
  const disableOptimize = optimizerLoading || optimizerRunning || filteredFileIds.length === 0;

  return (
    <div className="panel" style={{ marginTop: 8 }}>
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Riskfolio-style Optimizer</h3>
        {activeBadge}
      </div>
      <p className="text-muted small" style={{ marginTop: 4 }}>
        Configure mean-risk optimisation parameters and submit the current selection to the Riskfolio backend.
      </p>

      <div className="card" style={{ marginTop: 10 }}>
        <div className="tabs" style={{ borderBottom: 'none', gap: 6 }}>
          <button
            type="button"
            className={`tab ${riskfolioMode === 'mean-risk' ? 'tab-active' : ''}`}
            onClick={() => setRiskfolioMode('mean-risk')}
          >
            Mean-Risk
          </button>
          <button type="button" className="tab" style={{ opacity: 0.5 }} disabled>
            Risk Parity (coming soon)
          </button>
          <button type="button" className="tab" style={{ opacity: 0.5 }} disabled>
            Hierarchical (coming soon)
          </button>
        </div>
      </div>

      {riskfolioMode === 'mean-risk' ? (
        <>
          <div className="grid-2" style={{ marginTop: 12 }}>
            <div className="card">
              <label className="field-label" htmlFor="objective">Objective</label>
              <select id="objective" className="input" value={meanRiskObjective} onChange={(event) => setMeanRiskObjective(event.target.value)}>
                <option value="MaxRet">Maximum Return</option>
                <option value="MinRisk">Minimum Risk</option>
                <option value="Sharpe">Maximum Risk-Adjusted Return Ratio</option>
                <option value="Utility">Maximum Utility</option>
              </select>
              <label className="field-label" htmlFor="return-model" style={{ marginTop: 12 }}>Return Model</label>
              <select id="return-model" className="input" value={meanRiskReturnModel} onChange={(event) => setMeanRiskReturnModel(event.target.value)}>
                <option value="arithmetic">Mean Historical Return (Arithmetic)</option>
              </select>
              <label className="field-label" htmlFor="risk-measure" style={{ marginTop: 12 }}>Risk Measure</label>
              <select id="risk-measure" className="input" value={meanRiskRiskMeasure} onChange={(event) => setMeanRiskRiskMeasure(event.target.value)}>
                <option value="MV">Variance / Standard Deviation</option>
                <option value="MSV">Semi-Variance</option>
                <option value="MAD">Mean Absolute Deviation (MAD)</option>
                <option value="GMD">Gini Mean Difference (GMD)</option>
                <option value="CVaR">CVaR</option>
              </select>
              <label className="field-label" htmlFor="cov-method" style={{ marginTop: 12 }}>Covariance Method</label>
              <select id="cov-method" className="input" value={meanRiskCovariance} onChange={(event) => setMeanRiskCovariance(event.target.value)}>
                <option value="ledoit">Ledoit-Wolf</option>
              </select>
              <label className="field-label" htmlFor="mu-method" style={{ marginTop: 12 }}>Mean Return Estimator</label>
              <select
                id="mu-method"
                className="input"
                value={meanRiskReturnEstimate}
                onChange={(event) => setMeanRiskReturnEstimate(event.target.value)}
              >
                <option value="hist">Sample Mean (Historical)</option>
              </select>
              <label className="field-label" htmlFor="bounds" style={{ marginTop: 12 }}>Weight Bounds (min / max)</label>
              <div className="flex gap-sm">
                <input
                  className="input"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  inputMode="decimal"
                  value={meanRiskMinBound}
                  onChange={(event) => setMeanRiskMinBound(sanitizeWeightInputString(event.target.value))}
                  onBlur={() => setMeanRiskMinBound((prev) => enforceWeightInputBounds(prev, '0'))}
                />
                <input
                  className="input"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  inputMode="decimal"
                  value={meanRiskMaxBound}
                  onChange={(event) => setMeanRiskMaxBound(sanitizeWeightInputString(event.target.value))}
                  onBlur={() => setMeanRiskMaxBound((prev) => enforceWeightInputBounds(prev, '1'))}
                />
              </div>
              <label className="field-label" htmlFor="budget" style={{ marginTop: 12 }}>Budget (sum of weights)</label>
              <input
                id="budget"
                className="input"
                type="number"
                step={0.1}
                value={meanRiskBudget}
                onChange={(event) => setMeanRiskBudget(Number(event.target.value))}
              />
              <label className="field-label" htmlFor="group-caps" style={{ marginTop: 12 }}>Group Caps (symbol=cap, comma separated)</label>
              <input
                id="group-caps"
                className="input"
                placeholder="e.g. MNQ=0.5, MES=0.4"
                value={meanRiskSymbolCaps}
                onChange={(event) => setMeanRiskSymbolCaps(event.target.value)}
              />
              <label className="field-label" htmlFor="strategy-caps" style={{ marginTop: 12 }}>Strategy Caps (strategy=cap, comma separated)</label>
              <input
                id="strategy-caps"
                className="input"
                placeholder="e.g. trend=0.6, mean_revert=0.4"
                value={meanRiskStrategyCaps}
                onChange={(event) => setMeanRiskStrategyCaps(event.target.value)}
              />
            </div>

            <div className="card">
              <label className="field-label" htmlFor="rf">
                Risk-Free Rate (annual %)
                <span className="text-muted small" style={{ marginLeft: 6 }}>
                  (Converted to daily % before Riskfolio uses it)
                </span>
              </label>
              <input
                id="rf"
                className="input"
                type="number"
                step={0.05}
                value={meanRiskRiskFree}
                onChange={(event) => setMeanRiskRiskFree(Number(event.target.value))}
              />
              <label className="field-label" htmlFor="risk-aversion" style={{ marginTop: 12 }}>Risk Aversion (Utility only)</label>
              <input
                id="risk-aversion"
                className="input"
                type="number"
                step={0.1}
                value={meanRiskRiskAversion}
                onChange={(event) => setMeanRiskRiskAversion(Number(event.target.value))}
              />
              <label className="field-label" htmlFor="alpha" style={{ marginTop: 12 }}>Tail Probability (alpha)</label>
              <input
                id="alpha"
                className="input"
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={meanRiskAlpha}
                onChange={(event) => setMeanRiskAlpha(Number(event.target.value))}
              />
              <div className="text-muted small" style={{ marginTop: 4 }}>
                Example: 0.05 corresponds to a 95% confidence level.
              </div>
              <label className="field-label" htmlFor="max-risk" style={{ marginTop: 12 }}>Max Risk (selected measure, decimal)</label>
              <input
                id="max-risk"
                className="input"
                type="number"
                value={meanRiskMaxRisk}
                onChange={(event) => setMeanRiskMaxRisk(event.target.value)}
              />
              <label className="field-label" htmlFor="min-return" style={{ marginTop: 12 }}>Minimum Return (decimal)</label>
              <input
                id="min-return"
                className="input"
                type="number"
                value={meanRiskMinReturn}
                onChange={(event) => setMeanRiskMinReturn(event.target.value)}
              />
              <label className="field-label" htmlFor="turnover" style={{ marginTop: 12 }}>Turnover Limit (decimal)</label>
              <input
                id="turnover"
                className="input"
                type="number"
                value={meanRiskTurnover}
                onChange={(event) => setMeanRiskTurnover(event.target.value)}
              />
              <label className="field-label" htmlFor="frontier-points" style={{ marginTop: 12 }}>Efficient Frontier Points</label>
              <input
                id="frontier-points"
                className="input"
                type="number"
                min={0}
                step={1}
                value={meanRiskFrontierPoints}
                onChange={(event) => setMeanRiskFrontierPoints(Number(event.target.value))}
              />
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <strong>Per-Asset Weight Overrides</strong>
            <div className="text-muted small" style={{ marginTop: 4 }}>
              Leave fields blank to use the default min/max bounds.
            </div>
            {filteredFileIds.length ? (
              <div className="table-wrapper" style={{ marginTop: 10 }}>
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th>Asset</th>
                      <th>Min</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredFileIds.map((fileId) => {
                      const override = meanRiskOverrides[fileId];
                      const label = fileLabelMap[fileId] || fileId;
                      return (
                        <tr key={fileId}>
                          <td>{label}</td>
                          <td>
                            <input
                              type="number"
                              className="input"
                              style={{ maxWidth: 110 }}
                              min={0}
                              max={1}
                              step={0.05}
                              inputMode="decimal"
                              value={override?.min ?? ''}
                              placeholder={String(minWeightBound)}
                              onChange={(event) => handleOverrideChange(fileId, 'min', event.target.value)}
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              className="input"
                              style={{ maxWidth: 110 }}
                              min={0}
                              max={1}
                              step={0.05}
                              inputMode="decimal"
                              value={override?.max ?? ''}
                              placeholder={String(maxWeightBound)}
                              onChange={(event) => handleOverrideChange(fileId, 'max', event.target.value)}
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="placeholder-text" style={{ marginTop: 10 }}>Add files to enable overrides.</div>
            )}
            <div className="text-muted small" style={{ marginTop: 8 }}>
              Overrides are enforced alongside symbol/strategy caps and must sum to the configured budget.
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <strong>Optimize</strong>
            <div className="flex gap-md" style={{ marginTop: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <button type="button" className="button" disabled={disableOptimize} onClick={handleOptimize}>
                {optimizerRunning ? 'Optimizing...' : optimizerLoading ? 'Submitting...' : 'Optimize'}
              </button>
              <div className="text-muted small">Status: {statusLabel}</div>
              <div
                style={{ flex: 1, minWidth: 180, background: '#1f2f4a', borderRadius: 6, height: 10, overflow: 'hidden' }}
              >
                <div style={{ width: `${progressWidth}%`, background: '#4cc3ff', height: '100%' }} />
              </div>
            </div>
            {optimizerError ? (
              <div className="text-error small" style={{ marginTop: 8 }}>{optimizerError}</div>
            ) : (
              <div className="text-muted small" style={{ marginTop: 8 }}>
                {optimizerResult ? 'Latest optimizer results are shown below.' : 'Run the optimizer to populate results.'}
              </div>
            )}
          </div>

          <div className="grid-2" style={{ marginTop: 16 }}>
            <div className="card">
              <strong>Allocation summary</strong>
              <div className="grid-2" style={{ marginTop: 12 }}>
                <div className="metric-card">
                  <span className="text-muted small">Expected Return</span>
                  <strong>{summary ? formatPercent(summary.expected_return) : '--'}</strong>
                </div>
                <div className="metric-card">
                  <span className="text-muted small">Risk</span>
                  <strong>{summary ? formatPercent(summary.risk) : '--'}</strong>
                </div>
                <div className="metric-card">
                  <span className="text-muted small">Sharpe</span>
                  <strong>{summary ? summary.sharpe.toFixed(2) : '--'}</strong>
                </div>
                <div className="metric-card">
                  <span className="text-muted small">Max Drawdown</span>
                  <strong>{summary ? formatPercent(summary.max_drawdown) : '--'}</strong>
                </div>
              </div>
              <div className="text-muted small" style={{ marginTop: 8 }}>
                {summary ? (
                  <>Objective: {summary.objective} | Capital: {formatCurrency(summary.capital)}</>
                ) : (
                  'Run optimizer to calculate allocation metrics.'
                )}
              </div>
            </div>

            <div className="card">
              <strong>Contract sizing (margin aware)</strong>
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Asset</th>
                    <th>Contracts</th>
                    <th>Notional</th>
                    <th>Margin</th>
                  </tr>
                </thead>
                <tbody>
                  {contractRows.length ? (
                    contractRows.map((row) => {
                      const allocation = weightRows.find((weight) => weight.asset === row.asset);
                      return (
                        <tr key={row.asset}>
                          <td>{allocation?.label ?? row.asset}</td>
                          <td>{row.contracts.toFixed(2)}</td>
                          <td>{formatCurrency(row.notional)}</td>
                          <td>{row.margin !== undefined && row.margin !== null ? formatCurrency(row.margin) : '--'}</td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={4}>Run optimizer to calculate sizing.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <strong>Weights</strong>
            <table className="compact-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Weight</th>
                  <th>Contracts</th>
                  <th>Margin / Contract</th>
                </tr>
              </thead>
              <tbody>
                {weightRows.length ? (
                  weightRows.map((row) => (
                    <tr key={row.asset}>
                      <td>{row.label}</td>
                      <td>{formatPercent(row.weight)}</td>
                      <td>{row.contracts.toFixed(2)}</td>
                      <td>
                        {row.margin_per_contract !== undefined && row.margin_per_contract !== null
                          ? formatCurrency(row.margin_per_contract)
                          : '--'}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4}>Run optimizer to populate weights.</td>
                  </tr>
                )}
              </tbody>
            </table>
            <div className="text-muted small" style={{ marginTop: 8 }}>
              Weights reflect the most recent optimization run.
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <strong>Suggested Contracts</strong>
            <div className="flex gap-md" style={{ alignItems: 'center', marginTop: 10, flexWrap: 'wrap' }}>
              <label className="field-label" htmlFor="riskfolio-equity" style={{ margin: 0 }}>Account Equity</label>
              <input
                id="riskfolio-equity"
                className="input"
                type="number"
                min={0}
                step={100}
                value={riskfolioContractEquity}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  setRiskfolioContractEquity(Number.isFinite(next) ? next : 0);
                }}
                style={{ maxWidth: 200 }}
              />
              <button
                type="button"
                className="button"
                disabled={!canApplyRiskfolio}
                onClick={handleApplyRiskfolioContracts}
              >
                Apply Suggested Contracts
              </button>
            </div>
            <div className="text-muted small" style={{ marginTop: 10 }} role="status" aria-live="polite">
              {riskfolioApplyMessage ||
                (canApplyRiskfolio
                  ? 'Adjust the account equity to sandbox suggested contract counts before applying.'
                  : 'Run the optimizer to populate suggested contracts.')}
            </div>
            <div className="table-wrapper" style={{ marginTop: 10 }}>
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Asset</th>
                    <th>Optimized Weight</th>
                    <th>Suggested Gap</th>
                    <th>Current Gap</th>
                    <th>Margin / Contract</th>
                    <th>Suggested Contracts</th>
                    <th>Suggested Margin</th>
                    <th>Current Contracts</th>
                    <th>Current Margin</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {riskfolioContracts.rows.length ? (
                    <>
                      {riskfolioContracts.rows.map((row) => (
                        <tr key={row.asset}>
                          <td>{row.label}</td>
                          <td>{formatPercent(row.weight)}</td>
                          <td>{formatRiskfolioGap(row.suggestedGap)}</td>
                          <td>{formatRiskfolioGap(row.currentGap)}</td>
                          <td>{row.marginPerContract > 0 ? formatCurrency(row.marginPerContract) : '--'}</td>
                          <td>{row.suggestedContracts.toLocaleString()}</td>
                          <td>{formatCurrency(row.suggestedMargin)}</td>
                          <td>{row.currentContracts.toLocaleString()}</td>
                          <td>{formatCurrency(row.currentMargin)}</td>
                          <td>{row.delta > 0 ? `+${row.delta}` : row.delta.toString()}</td>
                        </tr>
                      ))}
                      {riskfolioContracts.summary ? (
                        <tr>
                          <td>Totals</td>
                          <td />
                          <td>{formatRiskfolioGapSummary(riskfolioContracts.summary.maxSuggestedGap, riskfolioContracts.summary.avgSuggestedGap)}</td>
                          <td>{formatRiskfolioGapSummary(riskfolioContracts.summary.maxCurrentGap, riskfolioContracts.summary.avgCurrentGap)}</td>
                          <td />
                          <td>{riskfolioContracts.summary.suggestedContractsTotal.toLocaleString()}</td>
                          <td>{formatCurrency(riskfolioContracts.summary.suggestedMarginTotal)}</td>
                          <td>{riskfolioContracts.summary.currentContractsTotal.toLocaleString()}</td>
                          <td>{formatCurrency(riskfolioContracts.summary.currentMarginTotal)}</td>
                          <td>
                            {(
                              riskfolioContracts.summary.suggestedContractsTotal -
                              riskfolioContracts.summary.currentContractsTotal
                            ).toLocaleString()}
                          </td>
                        </tr>
                      ) : null}
                    </>
                  ) : (
                    <tr>
                      <td colSpan={10}>Run the optimizer to populate suggested contracts.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="text-muted small" style={{ marginTop: 8 }}>
              {riskfolioContracts.note}
            </div>
          </div>

          <div className="grid-2" style={{ marginTop: 16 }}>
            <div className="card">
              <strong>Frontier Explorer</strong>
              {frontierPoints.length ? (
                <>
                  <div className="text-muted small" style={{ marginTop: 6 }}>
                    Click any point to compare it against the optimized allocation.
                  </div>
                  <EfficientFrontierChart
                    points={frontierPoints}
                    selectedIndex={selectedFrontierIdx}
                    onSelect={handleFrontierSelect}
                  />
                  {selectedFrontierMetrics ? (
                    <div className="grid-2" style={{ marginTop: 12 }}>
                      <div className="metric-card">
                        <span className="text-muted small">Selected Expected Return</span>
                        <strong>{formatPercent(selectedFrontierMetrics.expected)}</strong>
                      </div>
                      <div className="metric-card">
                        <span className="text-muted small">Selected Risk</span>
                        <strong>{formatPercent(selectedFrontierMetrics.risk)}</strong>
                      </div>
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="placeholder-text">
                  Run optimizer with efficient frontier points &gt; 0 to populate the chart.
                </div>
              )}
            </div>
            <div className="card">
              <strong>Allocation Across Frontier</strong>
              {frontierAllocationSeries.length ? (
                <>
                  <div className="text-muted small" style={{ marginTop: 6 }}>
                    Hover to inspect how each asset shifts along the frontier, or click anywhere to set the comparison point.
                  </div>
                  <FrontierAllocationAreaChart
                    series={frontierAllocationSeries}
                    selectedIndex={selectedFrontierIdx}
                    onSelect={handleFrontierSelect}
                  />
                </>
              ) : (
                <div className="placeholder-text">
                  Run optimizer with multiple assets to inspect allocation drift along the frontier.
                </div>
              )}
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <strong>Weight Comparison vs Optimized Solution</strong>
            {frontierComparisonRows.length ? (
              <div className="table-wrapper" style={{ marginTop: 10 }}>
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th>Asset</th>
                      <th>Optimized</th>
                      <th>Frontier</th>
                      <th>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {frontierComparisonRows.map((row) => (
                      <tr key={row.asset}>
                        <td>{row.label}</td>
                        <td>{formatPercent(row.incumbent)}</td>
                        <td>{formatPercent(row.frontier)}</td>
                        <td>{row.delta > 0 ? `+${formatPercent(row.delta)}` : formatPercent(row.delta)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="placeholder-text">
                Select a point on the efficient frontier to compare against the optimized allocation.
              </div>
            )}
          </div>

          <div className="grid-2" style={{ marginTop: 16 }}>
            <div>
              {optimizerBacktestSeries.length ? (
                <EquityMultiChart
                  title="Backtested Portfolio Equity"
                  description="Daily rebalanced using optimizer weights versus an equal-weight baseline."
                  series={optimizerBacktestSeries}
                  height={320}
                />
              ) : (
                <div className="card">
                  <strong>Backtested Portfolio Equity</strong>
                  <div className="placeholder-text">Run the optimizer to generate the backtest overlay.</div>
                </div>
              )}
            </div>
            <div className="card">
              <strong>Asset Correlation</strong>
              {optimizerCorrelationData ? (
                <>
                  <CorrelationHeatmap data={optimizerCorrelationData} />
                  <div className="text-muted small" style={{ marginTop: 8 }}>
                    Correlation matrix derived from the Riskfolio input returns (akin to plot_clusters).
                  </div>
                </>
              ) : (
                <div className="placeholder-text">
                  Need at least two optimized assets to render the correlation heatmap.
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="text-muted small">This optimisation method is coming soon. Switch to Mean-Risk to configure controls.</div>
        </div>
      )}
    </div>
  );
};


  const renderCta = () => (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>CTA-Style Report</h3>

        <div className="badge">Coming soon</div>

      </div>

      <p className="text-muted small" style={{ marginTop: 4 }}>

        This tab will eventually mirror the Dash CTA report (ROI summary, monthly/annual ROR tables, downloads). Until the
        backend is finalized, data is intentionally withheld.

      </p>

      <div className="card" style={{ marginTop: 12 }}>

        <div className="placeholder-text">

          {hasFiles

            ? 'CTA analytics will appear here once implemented.'

            : 'Upload files now so CTA analytics can run when the feature launches.'}

        </div>

      </div>

      <div className="card" style={{ marginTop: 14 }}>

        <strong>Download actions</strong>

        <div className="flex gap-md" style={{ marginTop: 10, flexWrap: 'wrap' }}>

          <button className="button" type="button" disabled>

            Export monthly ROR (CSV)

          </button>

          <button className="button" type="button" disabled>

            Export annual ROR (CSV)

          </button>

          <button className="button" type="button" disabled>

            Export metrics (CSV/Parquet)

          </button>

        </div>

        <div className="text-muted small" style={{ marginTop: 8 }}>

          Hooks mirror Dash exports; backend wiring will route to /api/v1/export once implemented.

        </div>

      </div>

    </div>

  );



  const renderPortfolioDrawdown = () => {

    if (!hasFiles) {
      return renderUploadPlaceholder('portfolio drawdown');
    }

    return (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Portfolio Drawdown</h3>

        {activeBadge}

      </div>

      <div className="card" style={{ marginTop: 12 }}>

        <strong>Plot lines</strong>

        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>

          {[...drawdownLines.perFile.map((s) => s.name), 'Portfolio'].map((name) => {

            const active = plotDrawdownEnabled[name] ?? true;

            return (

              <button

                key={name}

                type="button"

                className={`chip ${active ? 'chip-active' : ''}`}

                onClick={() => setPlotDrawdownEnabled((prev) => ({ ...prev, [name]: !active }))}

              >

                {name}

              </button>

            );

          })}

        </div>

      </div>



      <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>

        <EquityMultiChart

          title="Portfolio Drawdown ($)"

          series={[

            ...drawdownLines.perFile.filter((s) => plotDrawdownEnabled[s.name] !== false),

            ...(plotDrawdownEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: drawdownLines.portfolio }]),

          ]}

        />

        <EquityMultiChart

          title="Portfolio Drawdown (%)"

          series={[

            ...drawdownPercentLines.perFile.filter((s) => plotDrawdownEnabled[s.name] !== false),

            ...(plotDrawdownEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: drawdownPercentLines.portfolio }]),

          ]}

        />

      </div>

    </div>

    );

  };



  const renderIntradayDrawdown = () => {

    if (!hasFiles) {
      return renderUploadPlaceholder('intraday drawdown');
    }

    return (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Intraday Drawdown</h3>

        {activeBadge}

      </div>

      <div className="card" style={{ marginTop: 12 }}>

        {intradayDdQuery.isError ? (

          <div className="placeholder-text">{getQueryErrorMessage(intradayDdQuery.error)}</div>

        ) : intradayDdQuery.data ? (

          <>

            <SeriesChart title="Intraday Drawdown" series={intradayDdQuery.data} color="#f4c95d" />

            <div className="text-muted small">

              Points: {intradayDdQuery.data?.downsampled_count ?? intradayDdQuery.data?.portfolio.length ?? 0}

            </div>

          </>

        ) : (

          <div className="placeholder-text">No intraday drawdown data.</div>

        )}

      </div>

    </div>

    );

  };



  const renderMargin = () => {

    if (!hasFiles) {
      return renderUploadPlaceholder('margin insights');
    }

    return (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Margin</h3>

        {activeBadge}

      </div>

      <div className="text-muted small" style={{ marginTop: 6 }}>

        Margin views honor filters (symbols/intervals/strategies), contract multipliers, and date range. Setting contracts to zero or toggling off filters will exclude that file from the chart and portfolio line.

      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <strong>Exposure views</strong>
        <div className="flex gap-md" style={{ marginTop: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <label className="field-label" htmlFor="exposure-debug" style={{ margin: 0 }}>
            Show debug exposure views
          </label>
          <input
            id="exposure-debug"
            type="checkbox"
            checked={showExposureDebug}
            onChange={(event) => setShowExposureDebug(event.target.checked)}
          />
        </div>
        <div className="text-muted small" style={{ marginTop: 8 }}>
          Enables per-file, per-symbol, and portfolio-step exposure series alongside daily max.
        </div>
      </div>



      <div className="card" style={{ marginTop: 12 }}>

        <strong>Plot lines</strong>

        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>

          {[...purchasingPowerLines.perFile.map((s) => s.name), 'Portfolio'].map((name) => {

            const active = plotMarginEnabled[name] ?? true;

            return (

              <button

                key={name}

                type="button"

                className={`chip ${active ? 'chip-active' : ''}`}

                onClick={() => setPlotMarginEnabled((prev) => ({ ...prev, [name]: !active }))}

              >

                {name}

              </button>

            );

          })}

        </div>

      </div>



      <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>

        <EquityMultiChart

          title="Purchasing Power ($)"

          description="Purchasing Power = Starting Balance + Portfolio P&L - Initial Margin Used"

          series={[

            ...purchasingPowerLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),

            ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: purchasingPowerLines.portfolio }]),

          ]}

        />

        <EquityMultiChart

          title="Purchasing Power Drawdown ($)"

          description="Purchasing Power Drawdown (dollars from peak)"

          series={[

            ...purchasingPowerDrawdownLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),

            ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: purchasingPowerDrawdownLines.portfolio }]),

            ...(purchasingPowerDrawdownReferenceLine.length

              ? [{ name: 'Account Equity', points: purchasingPowerDrawdownReferenceLine }]

              : []),

          ]}

        />
        <EquityMultiChart
          title="Start-Date Safety ($)"
          description="Worst future purchasing power if you started on that date"
          series={
            plotMarginEnabled['Portfolio'] === false
              ? []
              : [{ name: 'Portfolio', points: startDateSafetyLine }]
          }
        />

        {showExposureDebug ? (
          <>
            <EquityMultiChart
              title="Initial Margin Used (Per File, Stepwise)"
              description="Initial Margin Used = |net_sym| x IM(sym)"
              series={[
                ...marginPerFileLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
                ...(plotMarginEnabled['Portfolio'] === false
                  ? []
                  : [{ name: 'Portfolio', points: marginPerFileLines.portfolio }]),
              ]}
            />
            <EquityMultiChart
              title="Initial Margin Used (Per Symbol, Stepwise)"
              description="Initial Margin Used = |net_sym| x IM(sym)"
              series={[
                ...marginPerSymbolLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
                ...(plotMarginEnabled['Portfolio'] === false
                  ? []
                  : [{ name: 'Portfolio', points: marginPerSymbolLines.portfolio }]),
              ]}
            />
            <EquityMultiChart
              title="Initial Margin Used (Portfolio Stepwise)"
              description="Initial Margin Used = |net_sym| x IM(sym)"
              series={plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: marginPortfolioStepLines.portfolio }]}
            />
            <EquityMultiChart
              title="Initial Margin Used (Portfolio Daily Max)"
              description="Initial Margin Used = |net_sym| x IM(sym)"
              series={plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: marginLines.portfolio }]}
            />
            <EquityMultiChart
              title="Net Contracts (Per File, Stepwise)"
              description="Net contracts over time (per file and portfolio)"
              series={[
                ...netposPerFileLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
                ...(plotMarginEnabled['Portfolio'] === false
                  ? []
                  : [{ name: 'Portfolio', points: netposPerFileLines.portfolio }]),
              ]}
            />
            <EquityMultiChart
              title="Net Contracts (Per Symbol, Stepwise)"
              description="Net contracts over time (per symbol and portfolio)"
              series={[
                ...netposPerSymbolLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
                ...(plotMarginEnabled['Portfolio'] === false
                  ? []
                  : [{ name: 'Portfolio', points: netposPerSymbolLines.portfolio }]),
              ]}
            />
            <EquityMultiChart
              title="Net Contracts (Portfolio Stepwise)"
              description="Net contracts over time (portfolio stepwise)"
              series={plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: netposPortfolioStepLines.portfolio }]}
            />
            <EquityMultiChart
              title="Net Contracts (Portfolio Daily Max)"
              description="Net contracts over time (portfolio daily max)"
              series={plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: netposLines.portfolio }]}
            />
          </>
        ) : (
          <>
            <EquityMultiChart
              title="Initial Margin Used ($)"
              description="Initial Margin Used = |net_sym| x IM(sym)"
              series={[
                ...marginLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
                ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: marginLines.portfolio }]),
              ]}
            />
            <EquityMultiChart
              title="Net Contracts"
              description="Net contracts over time (per file and portfolio)"
              series={[
                ...netposLines.perFile.filter((s) => plotMarginEnabled[s.name] !== false),
                ...(plotMarginEnabled['Portfolio'] === false ? [] : [{ name: 'Portfolio', points: netposLines.portfolio }]),
              ]}
            />
          </>
        )}

      </div>

    </div>

    );

  };



  const renderHistogram = () => {

    if (!hasFiles) {
      return renderUploadPlaceholder('histogram analysis');
    }

    return (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Trade P/L Histogram</h3>

        {activeBadge}

      </div>

      <div className="card" style={{ marginTop: 12 }}>

        <strong>Plot lines</strong>

        <div className="chips" style={{ marginTop: 10, flexWrap: 'wrap' }}>

          {histogramData.map((h) => h.name).map((name) => {

            const active = plotHistogramEnabled[name] ?? true;

            return (

              <button

                key={name}

                type="button"

                className={`chip ${active ? 'chip-active' : ''}`}

                onClick={() => setPlotHistogramEnabled((prev) => ({ ...prev, [name]: !active }))}

              >

                {name}

              </button>

            );

          })}

        </div>

      </div>



      {(() => {

        const activeHists = histogramData.filter((h) => plotHistogramEnabled[h.name] !== false);

        const baseBuckets = activeHists[0]?.buckets ?? [];

        const aggregatedBuckets = baseBuckets.map((bucket) => {

          const totalCount = activeHists.reduce((sum, hist) => {

            const match = hist.buckets.find(

              (candidate) =>

                candidate.start_value === bucket.start_value && candidate.end_value === bucket.end_value,

            );

            return sum + (match?.count ?? 0);

          }, 0);

          return { ...bucket, count: totalCount };

        });

        const portfolioBucketsDollar = aggregatedBuckets.map((bucket) => ({

          ...bucket,

          bucket: formatHistogramDollarRange(bucket.start_value, bucket.end_value),

        }));

        const portfolioBucketsPercent = aggregatedBuckets.map((bucket) => ({

          ...bucket,

          bucket: formatHistogramPercentRange(bucket.start_value, bucket.end_value, accountEquity),

        }));

        return (

          <div style={{ marginTop: 12, display: 'grid', gap: 12 }}>

            <div className="card">

              <strong>Portfolio Histogram ($)</strong>

              <div className="text-muted small" style={{ marginTop: 4 }}>

                Portfolio distribution; toggles exclude selected files from the sum.

              </div>

              <HistogramChart histogram={{ label: 'Portfolio Histogram ($)', buckets: portfolioBucketsDollar }} />

              <div className="text-muted small" style={{ marginTop: 8 }}>Buckets: {portfolioBucketsDollar.length}</div>

            </div>

            <div className="card">

              <strong>Portfolio Histogram (%)</strong>

              <div className="text-muted small" style={{ marginTop: 4 }}>

                Portfolio return distribution; toggles exclude selected files from the sum.

              </div>

              <HistogramChart histogram={{ label: 'Portfolio Histogram (%)', buckets: portfolioBucketsPercent }} />

              <div className="text-muted small" style={{ marginTop: 8 }}>Buckets: {portfolioBucketsPercent.length}</div>

            </div>

          </div>

        );

      })()}

    </div>

    );

  };



  const renderMetrics = () => {

    if (!hasFiles) {
      return renderUploadPlaceholder('metrics');
    }

    return (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Metrics</h3>

        <div className="flex" style={{ gap: 8, alignItems: 'center' }}>

          <button

            type="button"

            className="button"

            onClick={handleMetricsComputation}

            disabled={!canQueryData || metricsQuery.isFetching}

          >

            {metricsQuery.isFetching ? 'Computing...' : metricsRequested ? 'Refresh Metrics' : 'Compute Metrics'}

          </button>

          {metricsRequested && activeBadge}

        </div>

      </div>

      <div className="text-muted small" style={{ marginTop: 6 }}>

        Metrics are computed on demand using the active filters, overrides, and account equity.

      </div>

      <div style={{ marginTop: 12 }}>

        {!metricsRequested ? (

          <div className="placeholder-text">Click "Compute Metrics" to generate the table.</div>

        ) : metricsQuery.isError ? (

          <div className="placeholder-text">{getQueryErrorMessage(metricsQuery.error)}</div>

        ) : metricsBlocks.length ? (

          <MetricsGrid blocks={metricsBlocks} />

        ) : metricsQuery.isFetching ? (

          <div className="placeholder-text">Computing metrics...</div>

        ) : (

          <div className="placeholder-text">No metrics returned yet.</div>

        )}

      </div>

    </div>

    );

  };



  const renderAllocator = () => (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Allocator</h3>

        {activeBadge}

      </div>

      <div className="card" style={{ marginTop: 12 }}>

        <div className="text-muted small">Allocator UI placeholder to mirror the legacy tab (sizing, cash, contracts).</div>

      </div>

    </div>

  );



  const renderSettings = () => (

    <div className="panel" style={{ marginTop: 8 }}>

      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>

        <h3 className="section-title" style={{ margin: 0 }}>Settings</h3>

        {activeBadge}

      </div>

      <div className="card" style={{ marginTop: 12 }}>

        <div className="text-muted small">

          Selection filters live in the left rail (symbols, timeframes, strategies, direction, date range, spike toggle). This tab will

          mirror the legacy settings layout as needed.

        </div>

      </div>

    </div>

  );



  const renderIngest = () => (

    <div className="panel" style={{ marginTop: 8 }}>

      <h3 className="section-title" style={{ margin: 0 }}>Load Trade Lists & Exports</h3>

      <p className="text-muted small" style={{ marginTop: 4 }}>

        Parity with the Dash "Load Trade Lists" and "Settings" tabs: upload area, included files grid, contract/margin overrides,

        account equity, spike toggle, downsample/export switches, and file metadata preview.

      </p>



      <div className="card" style={{ marginTop: 14 }}>

        <div className="upload-area">

          <label htmlFor={UPLOAD_INPUT_ID} style={{ cursor: 'pointer', display: 'block' }}>

            Drag & drop or select .xlsx files

          </label>

          <input

            id={UPLOAD_INPUT_ID}

            type="file"

            accept=".xlsx"

            multiple

            style={{ display: 'none' }}

            onChange={async (event) => {

              const files = event.target.files;

              if (!files || !files.length) return;

              const formData = new FormData();

              Array.from(files).forEach((file) => formData.append('files', file));

              try {

                setUploadStatus('Uploading...');

                const response = await uploadFiles(formData);

                setUploadStatus(`Upload job ${response.job_id} completed (${response.files.length} files ingested)`);

                const meta = await getSelectionMeta();
                setSelectionMeta(meta);
                setFilesMeta(meta.files);
                setActiveSelection((prev) => ({
                  ...prev,
                  ...deriveSelectionDefaults(meta, prev),
                }));

              } catch (error) {

                setUploadStatus('Upload failed');

                setErrorMessage(error instanceof Error ? error.message : 'Upload failed');

              } finally {

                if (event.target) {

                  event.target.value = '';

                }

              }

            }}

          />

        </div>

        <div className="text-muted small" style={{ marginTop: 6 }}>

          Uploads mirror Dash (multiple files, persistent list).

          {uploadStatus ? ` ${uploadStatus}` : ''}

        </div>

      </div>



      {filesMeta.length ? (

        <>

      <div className="grid-2" style={{ marginTop: 14 }}>

        <div className="card">

          <strong>Included files</strong>

          <div

            style={{

              display: 'grid',

              gridTemplateColumns: '1fr 120px 140px',

              gap: 10,

              alignItems: 'center',

              marginTop: 8,

            }}

          >

            <div className="text-muted small" />

            <button

              type="button"

              className="button"

              style={{ width: '100%' }}

              onClick={() => {

                const defaults = Object.fromEntries(availableFiles.map((file) => [file, 1])) as Record<string, number>;

                setActiveSelection((prev) => ({

                  ...prev,

                  contracts: defaults,

                  contractMultipliers: defaults,

                }));

              }}

            >

              Use default contracts (1)

            </button>

            <button
              type="button"
              className="button"
              style={{ width: '100%' }}
              onClick={() => {
                const defaults = Object.fromEntries(
                  availableFiles.map((file) => [file, marginDefaultsByFile[file] ?? FALLBACK_MARGIN]),
                ) as Record<string, number>;
                setActiveSelection((prev) => ({
                  ...prev,
                  margins: defaults,
                  marginOverrides: defaults,
                }));
              }}
            >
              Use default margin
            </button>
          </div>

          <div className="file-rows" style={{ marginTop: 10, display: 'grid', gap: 10 }}>

            {availableFiles.map((fileId) => {

              const label = fileLabelMap[fileId] || fileId;

              const active = activeSelection.files.includes(fileId);

              const contractValue =

                (activeSelection.contractMultipliers ?? activeSelection.contracts)?.[fileId] ?? 1;

              const marginValue =
                (activeSelection.marginOverrides ?? activeSelection.margins)?.[fileId] ??
                marginDefaultsByFile[fileId] ??
                '';
              const filteredIn = filteredFileSet.has(fileId);

              const filteredOut = active && !filteredIn;

              return (

                <div

                  key={fileId}

                  className="card"

                  style={{

                    padding: '10px 12px',

                    display: 'grid',

                    gridTemplateColumns: '1fr 120px 140px',

                    gap: 10,

                    alignItems: 'center',

                  }}

                >

                  <button

                    type="button"

                    className={`chip ${active ? 'chip-active' : ''}`}

                    title={filteredOut ? 'Excluded by current filters' : undefined}

                    onClick={() =>

                      setActiveSelection((prev) => ({

                        ...prev,

                        files: active ? prev.files.filter((f) => f !== fileId) : [...prev.files, fileId],

                      }))

                    }

                    style={{ justifyContent: 'flex-start', opacity: filteredOut ? 0.4 : 1 }}

                  >

                    {label}

                  </button>

                  <input

                    type="number"

                    min={0}

                    step={1}

                    className="input"

                    style={{ opacity: filteredOut ? 0.5 : 1 }}

                    value={contractValue}

                    onChange={(event) => {

                      const next = Number(event.target.value);

                      setActiveSelection((prev) => {

                        const nextValue = Number.isNaN(next) ? 0 : next;

                        const nextContracts = {

                          ...(prev.contractMultipliers ?? prev.contracts ?? {}),

                          [fileId]: nextValue,

                        };

                        return {

                          ...prev,

                          contracts: nextContracts,

                          contractMultipliers: nextContracts,

                        };

                      });

                    }}

                    placeholder="Contracts"

                  />

                  <input
                    type="number"
                    step={100}
                    className="input"

                    style={{ opacity: filteredOut ? 0.5 : 1 }}

                    value={marginValue}

                    onChange={(event) => {

                      const next = Number(event.target.value);

                      setActiveSelection((prev) => {

                        const nextValue = Number.isNaN(next) ? 0 : next;

                        const nextMargins = { ...(prev.marginOverrides ?? prev.margins ?? {}), [fileId]: nextValue };

                        return {

                          ...prev,

                          margins: nextMargins,

                          marginOverrides: nextMargins,

                        };

                      });

                    }}

                    placeholder="Margin $/contract"

                  />

                </div>

              );

            })}

          </div>

          <div className="text-muted small" style={{ marginTop: 8 }}>Toggle files and edit contracts/margin per file.</div>

        </div>

      </div>



      <div className="grid-2" style={{ marginTop: 14 }}>

        <div className="card">

          <strong>Account equity</strong>

          <input
            className="input"
            type="number"
            step={1000}
            value={accountEquity}
            onChange={(event) => {
              const next = Number(event.target.value);
              setActiveSelection((prev) => ({
                ...prev,
                accountEquity: Number.isNaN(next) ? 0 : next,
              }));
            }}
            style={{ marginTop: 8, maxWidth: 240 }}
          />
          <div className="text-muted small" style={{ marginTop: 8 }}>

            Mirrors Dash account equity input (used for purchasing power and the margin tab).

          </div>

        </div>

      </div>



      <div style={{ marginTop: 12 }}>

        <SelectionControls

          selection={activeSelection}

          availableFiles={availableFiles}

          fileLabelMap={fileLabelMap}

          matchingFileCount={filteredFileIds.length}

          onChange={setActiveSelection}

        />

      </div>



      <div className="card" style={{ marginTop: 14 }}>

        <strong>Metadata preview</strong>

        <table className="compact-table">

          <thead>

            <tr>

              <th>File</th>

              <th>Symbols</th>

              <th>Intervals</th>

              <th>Strategies</th>

              <th>Date range</th>

            </tr>

          </thead>

          <tbody>

            {filesMeta.map((file) => (

              <tr key={file.file_id}>

                <td>{file.filename}</td>

                <td>{file.symbols.join(', ')}</td>

                <td>{file.intervals.join(', ')}</td>

                <td>{file.strategies.join(', ')}</td>

                <td>

                  {file.date_min || file.date_max ? `${file.date_min || '—'} to ${file.date_max || '—'}` : '—'}

                </td>

              </tr>

            ))}

          </tbody>

        </table>

        <div className="text-muted small" style={{ marginTop: 8 }}>

          Acts as the Dash helper showing parsed symbol/interval/strategy/date metadata per file.

        </div>

      </div>



      <div className="card" style={{ marginTop: 18 }}>

        <strong>Export options</strong>

        <div className="flex gap-md" style={{ marginTop: 10, alignItems: 'center', flexWrap: 'wrap' }}>

          <label className="field-label" htmlFor="downsample" style={{ margin: 0 }}>Downsample series</label>

          <input

            id="downsample"

            type="checkbox"

            checked={includeDownsample}

            onChange={(event) => setIncludeDownsample(event.target.checked)}

          />

          <label className="field-label" htmlFor="format" style={{ margin: 0 }}>Export format</label>

          <select

            id="format"

            className="input"

            value={exportFormat}

            onChange={(event) => setExportFormat(event.target.value as 'csv' | 'parquet')}

            style={{ maxWidth: 200 }}

          >

            <option value="csv">CSV</option>

            <option value="parquet">Parquet</option>

          </select>

        </div>

        <div className="text-muted small" style={{ marginTop: 8 }}>Maps to /api/v1/export/trades and /api/v1/export/metrics.</div>

      </div>

        </>

      ) : (

        <div className="card" style={{ marginTop: 16 }}>

          <div className="placeholder-text">No files yet—use the upload tool above to ingest trade lists.</div>

        </div>

      )}

    </div>

  );



  const renderTab = () => {

    if (activeTab === 'summary') return renderSummary();

    if (activeTab === 'equity-curves') return renderEquityCurves();

    if (activeTab === 'correlations') return renderCorrelations();

    if (activeTab === 'riskfolio') return renderOptimizer();

    if (activeTab === 'cta-report') return renderCta();

    if (activeTab === 'load-trade-lists') return renderIngest();

    if (activeTab === 'metrics') return renderMetrics();

    if (activeTab === 'portfolio-drawdown') return renderPortfolioDrawdown();

    if (activeTab === 'margin') return renderMargin();

    if (activeTab === 'trade-pl-histogram') return renderHistogram();


    return renderEquityCurves();

  };



  return (

    <div className="page">

      {errorMessage && (

        <div className="panel" style={{ borderColor: '#ff6b6b', background: 'rgba(255, 107, 107, 0.08)' }}>

          <div className="text-muted small">

            {errorMessage}

            <button

              type="button"

              className="button"

              style={{ marginLeft: 10 }}

              onClick={() => setErrorMessage(null)}

            >

              Dismiss

            </button>

          </div>

        </div>

      )}

      <div className="panel" style={{ marginBottom: 12 }}>

        <div className="flex gap-sm" style={{ alignItems: 'center', justifyContent: 'space-between' }}>

          <div>

            <h1 style={{ margin: '6px 0 8px 0' }}>Futures Portfolio Dashboard</h1>

            <div className="text-muted small">

              {apiBase ? (

                <span>

                  <span className="status-dot" /> API base configured: {apiBase}

                </span>

              ) : (

                <span>

                  <span className="status-dot" style={{ background: '#ffcb6b' }} /> API base not configured; set NEXT_PUBLIC_API_BASE to load live data

                </span>

              )}

            </div>

          </div>

          <button

            type="button"

            className="button"

            onClick={() => {

              equityQuery.refetch();

              equityPctQuery.refetch();

              drawdownQuery.refetch();

              intradayDdQuery.refetch();

              netposQuery.refetch();

              marginQuery.refetch();
              marginPortfolioStepQuery.refetch();

              histogramQuery.refetch();

              if (metricsRequested) {
                metricsQuery.refetch();
              }
              if (showExposureDebug) {
                netposPerFileQuery.refetch();
                netposPerSymbolQuery.refetch();
                netposPortfolioStepQuery.refetch();
                marginPerFileQuery.refetch();
                marginPerSymbolQuery.refetch();
              }


          

            }}

            disabled={busy}

          >

            {busy ? 'Refreshing...' : 'Refresh'}

          </button>

        </div>

      </div>



      {apiMissing && (

        <div className="panel" style={{ borderColor: '#ffcb6b', background: 'rgba(255, 203, 107, 0.06)' }}>

          <div className="text-muted small">

            Live uploads, series, metrics, and histogram calls are disabled until NEXT_PUBLIC_API_BASE is set.

          </div>

        </div>

      )}



      <div className="panel" style={{ overflow: 'hidden' }}>

        <div className="tabs">

          {tabs.map((tab) => (

            <button

              key={tab.key}

              type="button"

              className={`tab ${activeTab === tab.key ? 'tab-active' : ''}`}

              onClick={() => setActiveTab(tab.key)}

            >

              {tab.label}

            </button>

          ))}

        </div>

        <div style={{ marginTop: 12 }}>{renderTab()}</div>

      </div>

    </div>

  );

}
