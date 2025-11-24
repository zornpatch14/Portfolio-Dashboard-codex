'use client';

import * as echarts from 'echarts';
import { useEffect, useRef } from 'react';
import { SeriesResponse } from '../lib/api';

type Props = {
  title: string;
  series: SeriesResponse;
  color?: string;
};

export default function SeriesChart({ title, series, color = '#4cc3ff' }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      grid: { left: 40, right: 20, top: 40, bottom: 30 },
      title: { text: title, left: 0, top: 0, textStyle: { color: '#e6edf7', fontSize: 14 } },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: series.points.map((p) => p.timestamp),
        axisLine: { lineStyle: { color: '#334b76' } },
        axisLabel: { color: '#a5b2c9' },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#334b76' } },
        splitLine: { lineStyle: { color: '#1f2f4a' } },
        axisLabel: { color: '#a5b2c9' },
      },
      series: [
        {
          name: series.label,
          type: 'line',
          smooth: true,
          showSymbol: false,
          data: series.points.map((p) => p.value),
          areaStyle: {
            opacity: 0.2,
            color,
          },
          lineStyle: { color, width: 2 },
        },
      ],
    });

    const resize = () => chart.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      chart.dispose();
    };
  }, [series, title, color]);

  return <div ref={ref} style={{ width: '100%', height: 260 }} />;
}
