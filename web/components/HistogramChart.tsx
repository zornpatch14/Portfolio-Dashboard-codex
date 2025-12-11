'use client';

import * as echarts from 'echarts';
import { useEffect, useRef } from 'react';
import { HistogramResponse } from '../lib/api';

type Props = {
  histogram: HistogramResponse;
};

export function HistogramChart({ histogram }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      grid: { left: 50, right: 20, top: 30, bottom: 70, containLabel: true },
      xAxis: {
        type: 'category',
        data: histogram.buckets.map((b) => b.bucket),
        axisLine: { lineStyle: { color: '#334b76' } },
        axisLabel: { color: '#a5b2c9', rotate: 45 },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#334b76' } },
        splitLine: { lineStyle: { color: '#1f2f4a' } },
        axisLabel: { color: '#a5b2c9' },
      },
      series: [
        {
          type: 'bar',
          name: histogram.label,
          data: histogram.buckets.map((b) => b.count),
          itemStyle: { color: '#8fe3c7' },
          barWidth: '60%',
        },
      ],
    });

    const resize = () => chart.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      chart.dispose();
    };
  }, [histogram]);

  return <div ref={ref} style={{ width: '100%', height: 260 }} />;
}
