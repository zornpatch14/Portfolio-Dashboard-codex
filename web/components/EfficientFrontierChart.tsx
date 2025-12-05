'use client';

import * as echarts from 'echarts';
import { useEffect, useRef } from 'react';

type FrontierPoint = {
  idx: number;
  risk: number;
  expectedReturn: number;
};

type Props = {
  points: FrontierPoint[];
  selectedIndex: number | null;
  onSelect?: (index: number) => void;
};

export function EfficientFrontierChart({ points, selectedIndex, onSelect }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current);
    const scatterData = points.map((point) => ({
      value: [point.risk, point.expectedReturn, point.idx],
      itemStyle: point.idx === selectedIndex ? { color: '#f4c95d' } : undefined,
    }));
    const lineData = [...points]
      .sort((a, b) => a.risk - b.risk)
      .map((point) => [point.risk, point.expectedReturn]);

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter(params: any) {
          const risk = params.value?.[0] ?? 0;
          const ret = params.value?.[1] ?? 0;
          return `Risk: ${(risk * 100).toFixed(2)}%<br/>Expected: ${(ret * 100).toFixed(2)}%`;
        },
      },
      grid: { left: 60, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'value',
        name: 'Risk',
        nameTextStyle: { color: '#a5b2c9' },
        axisLine: { lineStyle: { color: '#334b76' } },
        splitLine: { lineStyle: { color: '#1f2f4a' } },
        axisLabel: {
          color: '#a5b2c9',
          formatter: (value: number) => `${(value * 100).toFixed(1)}%`,
        },
      },
      yAxis: {
        type: 'value',
        name: 'Expected Return',
        nameTextStyle: { color: '#a5b2c9' },
        axisLine: { lineStyle: { color: '#334b76' } },
        splitLine: { lineStyle: { color: '#1f2f4a' } },
        axisLabel: {
          color: '#a5b2c9',
          formatter: (value: number) => `${(value * 100).toFixed(1)}%`,
        },
      },
      series: [
        {
          name: 'Frontier',
          type: 'line',
          data: lineData,
          showSymbol: false,
          smooth: true,
          lineStyle: { color: '#2f81f7', width: 1 },
          emphasis: { focus: 'series' },
          zlevel: 1,
        },
        {
          name: 'Portfolios',
          type: 'scatter',
          data: scatterData,
          symbolSize: (value: any) => (value?.[2] === selectedIndex ? 18 : 10),
          itemStyle: {
            color: '#54ffd0',
            opacity: 0.85,
          },
          emphasis: {
            scale: true,
          },
          zlevel: 2,
        },
      ],
    };

    chart.setOption(option);

    const handleClick = (params: any) => {
      const data = params?.data;
      const idx = Array.isArray(data) ? data[2] : data?.value?.[2];
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
  }, [points, selectedIndex, onSelect]);

  return <div ref={ref} style={{ width: '100%', height: 280 }} />;
}
