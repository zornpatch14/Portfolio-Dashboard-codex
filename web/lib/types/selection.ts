export type Selection = {
  name: string;
  files: string[];
  fileLabels?: Record<string, string>;
  symbols: string[];
  intervals: string[];
  strategies: string[];
  direction: string;
  start: string | null;
  end: string | null;
  contracts: Record<string, number>;
  margins: Record<string, number>;
  contractMultipliers?: Record<string, number>;
  marginOverrides?: Record<string, number>;
  spike: boolean;
  downsample?: boolean;
  accountEquity?: number | null;
};
