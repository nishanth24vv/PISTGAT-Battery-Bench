import React from 'react';
import { CellData } from './Canvas3D';
import { AlertTriangle, CheckCircle, Zap } from 'lucide-react';

interface CellHeatmapProps {
  cells: Record<string, CellData>;
  packImbalanceDelta: number;
  selectedCell: string | null;
  onSelectCell: (cellId: string | null) => void;
}

export const CellHeatmap: React.FC<CellHeatmapProps> = ({
  cells,
  packImbalanceDelta,
  selectedCell,
  onSelectCell
}) => {
  const cellEntries = Object.entries(cells);
  const deltaPercent = packImbalanceDelta * 100;

  // Imbalance severity
  let severityBadge = (
    <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
      <CheckCircle className="w-3.5 h-3.5 mr-1" /> Balanced (Delta &lt; 2%)
    </span>
  );
  if (deltaPercent >= 5.0) {
    severityBadge = (
      <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20 animate-pulse">
        <AlertTriangle className="w-3.5 h-3.5 mr-1" /> Critical Imbalance (BMS Balancing Active)
      </span>
    );
  } else if (deltaPercent >= 2.0) {
    severityBadge = (
      <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
        <Zap className="w-3.5 h-3.5 mr-1" /> Moderate Divergence
      </span>
    );
  }

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h3 className="text-base font-semibold text-slate-100">Cell-Level Imbalance Distribution</h3>
          <p className="text-xs text-slate-400">Inter-cell thermal and capacity gradients across series string</p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="text-right">
            <span className="text-xs text-slate-400 block">Pack Imbalance (ΔSOH)</span>
            <span className="text-sm font-bold text-amber-400">+{deltaPercent.toFixed(2)}%</span>
          </div>
          {severityBadge}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
        {cellEntries.map(([id, data]) => {
          const isSelected = selectedCell === id;
          const sohPct = (data.soh_pred * 100).toFixed(1);
          const dev = (data.soh_deviation * 100).toFixed(2);
          const devPositive = data.soh_deviation >= 0;

          // Color border based on SOH
          let borderTint = 'border-slate-800 hover:border-slate-700';
          let bgTint = 'bg-slate-950/60';
          if (data.soh_pred >= 0.90) {
            borderTint = isSelected ? 'border-emerald-400 ring-2 ring-emerald-500/30' : 'border-emerald-950/60 hover:border-emerald-700/60';
          } else if (data.soh_pred >= 0.80) {
            borderTint = isSelected ? 'border-yellow-400 ring-2 ring-yellow-500/30' : 'border-yellow-950/60 hover:border-yellow-700/60';
          } else {
            borderTint = isSelected ? 'border-red-400 ring-2 ring-red-500/30' : 'border-red-950/60 hover:border-red-700/60';
          }

          return (
            <button
              key={id}
              onClick={() => onSelectCell(isSelected ? null : id)}
              className={`p-3 rounded-xl border text-left transition-all duration-200 cursor-pointer ${borderTint} ${bgTint}`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="font-semibold text-slate-300 truncate max-w-[85px]">{id}</span>
                <span
                  className={`text-[10px] font-mono px-1 rounded ${
                    devPositive ? 'text-emerald-400 bg-emerald-500/10' : 'text-red-400 bg-red-500/10'
                  }`}
                >
                  {devPositive ? `+${dev}%` : `${dev}%`}
                </span>
              </div>
              <div className="text-lg font-bold text-slate-100 mb-1">{sohPct}%</div>
              <div className="flex justify-between items-center text-[10px] text-slate-400">
                <span>{data.temperature_c.toFixed(1)}°C</span>
                <span>{(data.resistance_ohm * 1000).toFixed(1)} mΩ</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
