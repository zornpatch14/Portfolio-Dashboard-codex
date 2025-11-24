'use client';

import * as echarts from 'echarts';
import { useEffect, useRef } from 'react';
import { CorrelationResponse } from '../lib/api';

type Props = {
  data: CorrelationResponse;
};

export function CorrelationHeatmap({ data }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const chart = echarts.init(ref.current);
    const seriesData = [] as { value: [number, number, number] }[];
    data.labels.forEach((_, row) => {
      data.labels.forEach((_, col) => {
        seriesData.push({ value: [col, row, parseFloat(data.matrix[row][col].toFixed(3))] });
      });
    });

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        position: 'top',
        formatter(params: any) {
          const [x, y, value] = params.data.value;
          return `${data.labels[y]} vs ${data.labels[x]}: <strong>${value}</strong>`;
        },
      },
      grid: { left: 80, right: 30, top: 30, bottom: 60 },
      xAxis: {
        type: 'category',
        data: data.labels,
        axisLine: { lineStyle: { color: '#334b76' } },
        axisLabel: { color: '#a5b2c9', rotate: 35 },
      },
      yAxis: {
        type: 'category',
        data: data.labels,
        axisLine: { lineStyle: { color: '#334b76' } },
        axisLabel: { color: '#a5b2c9' },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 12,
        inRange: {
          color: ['#ff8f6b', '#15213a', '#54ffd0'],
        },
        textStyle: { color: '#e6edf7' },
      },
      series: [
        {
          name: data.mode,
          type: 'heatmap',
          data: seriesData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.4)',
            },
          },
        },
      ],
    });

    const resize = () => chart.resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      chart.dispose();
    };
  }, [data]);

  return <div ref={ref} style={{ width: '100%', height: 380 }} />;
}
