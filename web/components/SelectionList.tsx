'use client';

import { selectionLabel, Selection } from '../lib/selections';

type Props = {
  selections: Selection[];
  active: Selection;
  onSelect: (selection: Selection) => void;
};

export function SelectionList({ selections, active, onSelect }: Props) {
  return (
    <div className="panel">
      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="section-title">Selections</h2>
        <span className="badge">Sample set</span>
      </div>
      <p className="text-muted small" style={{ marginTop: 0, marginBottom: 12 }}>
        Loaded from <code>tests/baseline/selections.json</code>. Click to drive mock or live API calls.
      </p>
      <div className="selection-list">
        {selections.map((selection) => (
          <button
            key={selection.name}
            type="button"
            className={`selection-item ${selection.name === active.name ? 'active' : ''}`}
            onClick={() => onSelect(selection)}
          >
            <div className="flex gap-sm" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <strong>{selection.name}</strong>
                <div className="text-muted small">{selection.files.length} files</div>
              </div>
              {selection.name === active.name ? <span className="badge">Selected</span> : null}
            </div>
            <div className="selection-meta text-muted small">
              <span>{selectionLabel(selection)}</span>
              <span>Direction: {selection.direction}</span>
              <span>Spike filter: {selection.spike ? 'On' : 'Off'}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
