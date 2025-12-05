'use client';

import * as echarts from 'echarts';
import { useEffect, useRef } from 'react';

type AllocationValue = {
  risk: number;
  weight: number;
  idx: number;
};

type AllocationSeries = {
  asset: string;
  values: AllocationValue[];
};

type Props = {
  series: AllocationSeries[];
  selectedIndex: number | null;
  onSelect?: (index: number) => void;
};

export function FrontierAllocationAreaChart({ series, selectedIndex, onSelect }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current);

    const chartSeries = series.map((line) => ({
      name: line.asset,
      type: 'line' as const,
      stack: 'weights',
      smooth: true,
      symbol: 'none',
      areaStyle: { opacity: 0.15 },
      emphasis: { focus: 'series' as const },
      data: line.values.map((value) => ({ value: [value.risk, value.weight, value.idx] })),
    }));

    const flattened = series.flatMap((line) => line.values);
    const selectedPoint =
      selectedIndex !== null ? flattened.find((value) => value.idx === selectedIndex) : undefined;

    if (chartSeries.length && selectedPoint) {
      chartSeries[0].markLine = {
        symbol: 'none',
        data: [{ xAxis: selectedPoint.risk }],
        silent: true,
        lineStyle: { color: '#f4c95d', type: 'dashed', width: 1.5 },
      };
    }

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'line' },
        formatter(params: any[]) {
          if (!params.length) return '';
          const risk = params[0].value?.[0] ?? 0;
          const rows = params
            .map((item) => `${item.marker} ${item.seriesName}: ${(item.value?.[1] * 100).toFixed(2)}%`)
            .join('<br/>');
          return `Risk: ${(risk * 100).toFixed(2)}%<br/>${rows}`;
        },
      },
      legend: {
        data: series.map((line) => line.asset),
        textStyle: { color: '#e6edf7' },
        top: 0,
      },
      grid: { left: 60, right: 20, top: 50, bottom: 40 },
      xAxis: {
        type: 'value',
        name: 'Risk',
        axisLine: { lineStyle: { color: '#334b76' } },
        splitLine: { lineStyle: { color: '#1f2f4a' } },
        axisLabel: {
          color: '#a5b2c9',
          formatter: (value: number) => `${(value * 100).toFixed(1)}%`,
        },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        name: 'Weight',
        axisLine: { lineStyle: { color: '#334b76' } },
        splitLine: { lineStyle: { color: '#1f2f4a' } },
        axisLabel: {
          color: '#a5b2c9',
          formatter: (value: number) => `${(value * 100).toFixed(0)}%`,
        },
      },
      series: chartSeries,
    });

    const handleClick = (params: any) => {
      const data = params?.data;
      const idx = data?.value?.[2];
      if (typeof idx === 'number') {
        onSelect?.(idx);
      }
    };

    chart.on('click', handleClick);
    const resize = () => chart.resize();
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      chart.off('click', handleClick);
      chart.dispose();
    };
  }, [series, selectedIndex, onSelect]);

  return <div ref={ref} style={{ width: '100%', height: 280 }} />;
}
