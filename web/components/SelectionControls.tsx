'use client';

import { useMemo } from 'react';
import type { Selection } from '../lib/types/selection';

type Props = {
  selection: Selection;
  availableFiles: string[];
  fileLabelMap: Record<string, string>;
  onChange: (selection: Selection) => void;
  matchingFileCount: number;
};

export function SelectionControls({ selection, availableFiles, fileLabelMap, onChange, matchingFileCount }: Props) {
  const toggleFile = (file: string) => {
    const exists = selection.files.includes(file);
    const files = exists ? selection.files.filter((f) => f !== file) : [...selection.files, file];
    onChange({ ...selection, files });
  };

  const parsed = useMemo(() => {
    const symbols = new Set<string>();
    const intervals = new Set<string>();
    const strategies = new Set<string>();
    availableFiles.forEach((file) => {
      const label = fileLabelMap[file] || file;
      const base = label.split('/').pop() || label;
      const parts = base.replace(/\.[^.]+$/, '').split('_');
      if (parts.length >= 4) {
        symbols.add(parts[1]);
        intervals.add(parts[2]);
        strategies.add(parts.slice(3).join('_'));
      }
    });
    return {
      symbols: Array.from(symbols).sort(),
      intervals: Array.from(intervals).sort((a, b) => Number(a) - Number(b)),
      strategies: Array.from(strategies).sort(),
    };
  }, [availableFiles]);

  return (
    <div className="panel">
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="section-title">Selection filters</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="badge">Interactive</span>
          <span className="text-muted small">Matching files: {matchingFileCount}</span>
        </div>
      </div>
      <p className="text-muted small" style={{ marginTop: 0, marginBottom: 16 }}>
        Adjust your selection to mirror the legacy Dash controls. Changes apply instantly and drive live API calls when
        configured.
      </p>

      <div className="grid-2">
        <div>
          <label className="field-label">Symbols</label>
          <div className="chips">
            {parsed.symbols.length ? (
              parsed.symbols.map((symbol) => {
                const active = selection.symbols.includes(symbol);
                return (
                  <button
                    key={symbol}
                    type="button"
                    className={`chip ${active ? 'chip-active' : ''}`}
                    onClick={() => {
                      const next = active
                        ? selection.symbols.filter((s) => s !== symbol)
                        : [...selection.symbols, symbol];
                      onChange({ ...selection, symbols: next });
                    }}
                  >
                    {symbol}
                  </button>
                );
              })
            ) : (
              <div className="text-muted small">No symbols detected yet.</div>
            )}
          </div>
        </div>
        <div>
          <label className="field-label">Intervals</label>
          <div className="chips">
            {parsed.intervals.length ? (
              parsed.intervals.map((interval) => {
                const active = selection.intervals.includes(interval);
                return (
                  <button
                    key={interval}
                    type="button"
                    className={`chip ${active ? 'chip-active' : ''}`}
                    onClick={() => {
                      const next = active
                        ? selection.intervals.filter((i) => i !== interval)
                        : [...selection.intervals, interval];
                      onChange({ ...selection, intervals: next });
                    }}
                  >
                    {interval}
                  </button>
                );
              })
            ) : (
              <div className="text-muted small">No intervals detected yet.</div>
            )}
          </div>
        </div>
        <div>
          <label className="field-label">Strategies</label>
          <div className="chips">
            {parsed.strategies.length ? (
              parsed.strategies.map((strategy) => {
                const active = selection.strategies.includes(strategy);
                return (
                  <button
                    key={strategy}
                    type="button"
                    className={`chip ${active ? 'chip-active' : ''}`}
                    onClick={() => {
                      const next = active
                        ? selection.strategies.filter((s) => s !== strategy)
                        : [...selection.strategies, strategy];
                      onChange({ ...selection, strategies: next });
                    }}
                  >
                    {strategy}
                  </button>
                );
              })
            ) : (
              <div className="text-muted small">No strategies detected yet.</div>
            )}
          </div>
        </div>
        <div className="flex gap-md" style={{ alignItems: 'flex-end' }}>
          <div style={{ flex: 1 }}>
            <label className="field-label" htmlFor="direction">Direction</label>
            <select
              id="direction"
              className="input"
              value={selection.direction}
              onChange={(event) => onChange({ ...selection, direction: event.target.value })}
            >
              <option value="All">All</option>
              <option value="Long">Long</option>
              <option value="Short">Short</option>
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label className="field-label" htmlFor="spike">Spike filter</label>
            <button
              type="button"
              className={`button ${selection.spike ? 'button-on' : 'button-off'}`}
              onClick={() => onChange({ ...selection, spike: !selection.spike })}
            >
              {selection.spike ? 'Spikes removed' : 'Spikes kept'}
            </button>
          </div>
        </div>
        <div className="flex gap-md" style={{ alignItems: 'flex-end' }}>
          <div style={{ flex: 1 }}>
            <label className="field-label" htmlFor="start">Start date</label>
            <input
              id="start"
              type="date"
              className="input"
              value={selection.start ?? ''}
              onChange={(event) => onChange({ ...selection, start: event.target.value || null })}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label className="field-label" htmlFor="end">End date</label>
            <input
              id="end"
              type="date"
              className="input"
              value={selection.end ?? ''}
              onChange={(event) => onChange({ ...selection, end: event.target.value || null })}
            />
          </div>
        </div>

      </div>

    </div>
  );
}
