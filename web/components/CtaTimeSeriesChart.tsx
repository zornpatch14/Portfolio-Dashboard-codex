'use client';

import * as echarts from 'echarts';
import { useEffect, useRef } from 'react';

type Point = {
  timestamp: string;
  value: number;
};

type Props = {
  title: string;
  points: Point[];
  type?: 'line' | 'bar';
  color?: string;
  height?: number;
  valueFormatter?: (value: number) => string;
};

export function CtaTimeSeriesChart({
  title,
  points,
  type = 'line',
  color = '#4cc3ff',
  height = 320,
  valueFormatter,
}: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current);
    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: unknown) => {
          if (typeof value !== 'number') return String(value);
          return valueFormatter ? valueFormatter(value) : value.toLocaleString();
        },
      },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
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
      title: { text: title, left: 0, top: 0, textStyle: { color: '#e6edf7', fontSize: 14 } },
      xAxis: {
        type: 'time',
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
          type,
          data: points.map((point) => [point.timestamp, point.value]),
          showSymbol: type === 'line' ? false : undefined,
          smooth: type === 'line',
          lineStyle: type === 'line' ? { color, width: 2 } : undefined,
          areaStyle: type === 'line' ? { opacity: 0.2, color } : undefined,
          itemStyle: type === 'bar' ? { color } : undefined,
          barMaxWidth: type === 'bar' ? 28 : undefined,
        },
      ],
    });

    const resize = () => chart.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      chart.dispose();
    };
  }, [points, title, type, color, height, valueFormatter]);

  return <div ref={ref} style={{ width: '100%', height }} />;
}
