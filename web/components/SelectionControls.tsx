'use client';

import { useMemo } from 'react';
import { Selection } from '../lib/selections';

function normalizeList(value: string) {
  return value
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
}

type Props = {
  selection: Selection;
  availableFiles: string[];
  onChange: (selection: Selection) => void;
};

export function SelectionControls({ selection, availableFiles, onChange }: Props) {
  const toggleFile = (file: string) => {
    const exists = selection.files.includes(file);
    const files = exists ? selection.files.filter((f) => f !== file) : [...selection.files, file];
    onChange({ ...selection, files });
  };

  const allSymbols = useMemo(() => normalizeList(selection.symbols.join(',')), [selection.symbols]);
  const allIntervals = useMemo(() => normalizeList(selection.intervals.join(',')), [selection.intervals]);
  const allStrategies = useMemo(() => normalizeList(selection.strategies.join(',')), [selection.strategies]);

  return (
    <div className="panel">
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="section-title">Selection filters</h2>
        <span className="badge">Interactive</span>
      </div>
      <p className="text-muted small" style={{ marginTop: 0, marginBottom: 16 }}>
        Adjust your selection to mirror the legacy Dash controls. Changes apply instantly and drive live API calls when
        configured.
      </p>

      <div className="grid-2">
        <div>
          <label className="field-label" htmlFor="symbols">Symbols (comma separated)</label>
          <input
            id="symbols"
            className="input"
            placeholder="e.g. MNQ, MES"
            value={allSymbols.join(', ')}
            onChange={(event) => onChange({ ...selection, symbols: normalizeList(event.target.value) })}
          />
        </div>
        <div>
          <label className="field-label" htmlFor="intervals">Intervals (comma separated)</label>
          <input
            id="intervals"
            className="input"
            placeholder="e.g. 15, 60"
            value={allIntervals.join(', ')}
            onChange={(event) => onChange({ ...selection, intervals: normalizeList(event.target.value) })}
          />
        </div>
        <div>
          <label className="field-label" htmlFor="strategies">Strategies (comma separated)</label>
          <input
            id="strategies"
            className="input"
            placeholder="All strategies"
            value={allStrategies.join(', ')}
            onChange={(event) => onChange({ ...selection, strategies: normalizeList(event.target.value) })}
          />
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

      <div style={{ marginTop: 18 }}>
        <div className="text-muted small" style={{ marginBottom: 8 }}>Files in selection</div>
        <div className="chips">
          {availableFiles.map((file) => {
            const active = selection.files.includes(file);
            return (
              <button
                key={file}
                type="button"
                className={`chip ${active ? 'chip-active' : ''}`}
                onClick={() => toggleFile(file)}
              >
                {file}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
