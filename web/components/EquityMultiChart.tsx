'use client';

import * as echarts from 'echarts';
import { useEffect, useRef } from 'react';
import { SeriesPoint } from '../lib/api';

type SeriesLine = {
  name: string;
  points: SeriesPoint[];
};

type Props = {
  title: string;
  series: SeriesLine[];
  description?: string;
  height?: number;
};

const palette = ['#4cc3ff', '#8fe3c7', '#ff8f6b', '#9f8bff', '#f4c95d', '#54ffd0', '#ffcb6b', '#6bdcff'];

export function EquityMultiChart({ title, series, description, height = 380 }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current);

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
      },
      legend: {
        data: series.map((s) => s.name),
        textStyle: { color: '#e6edf7' },
        top: 4,
      },
      grid: { left: 50, right: 20, top: 50, bottom: 40 },
      dataZoom: [
        {
          type: 'slider',
          height: 18,
          bottom: 10,
          borderColor: '#334b76',
          backgroundColor: 'rgba(30, 46, 73, 0.7)',
          handleStyle: { color: '#8fe3c7', borderColor: '#1b2c44' },
          textStyle: { color: '#a5b2c9' },
        },
        {
          type: 'inside',
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
      ],
      xAxis: {
        type: 'time',
        axisLine: { lineStyle: { color: '#334b76' } },
        axisLabel: { color: '#a5b2c9' },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#334b76' } },
        axisLabel: { color: '#a5b2c9' },
        splitLine: { lineStyle: { color: '#1f2f4a' } },
      },
      series: series.map((line, idx) => ({
        name: line.name,
        type: 'line',
        data: line.points.map((p) => [p.timestamp, p.value]),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: palette[idx % palette.length] },
      })),
    };

    chart.setOption(option);
    const resize = () => chart.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      chart.dispose();
    };
  }, [series]);

  return (
    <div className="card">
      <div className="flex" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
        <strong>{title}</strong>
      </div>
      {description ? (
        <div className="text-muted small" style={{ marginTop: 4 }}>
          {description}
        </div>
      ) : null}
      <div ref={ref} style={{ width: '100%', height }} />
    </div>
  );
}
