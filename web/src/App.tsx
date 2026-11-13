import React, { useState, useEffect, useRef } from 'react';
import { Canvas3D, CellData } from './components/Canvas3D';
import { CellHeatmap } from './components/CellHeatmap';
import { MetricsChart } from './components/MetricsChart';
import {
  Battery,
  Play,
  Pause,
  RotateCcw,
  Upload,
  Cpu,
  Activity,
  Layers,
  Thermometer,
  ShieldCheck,
  Github
} from 'lucide-react';

interface TelemetryPayload {
  metadata?: {
    dataset?: string;
    pack_configuration?: string;
    nominal_capacity_ah?: number;
    cells_count?: number;
  };
  benchmark_results?: Record<string, any>;
  timeline: Array<{
    cycle: number;
    pack_soh_mean: number;
    pack_imbalance_delta: number;
    max_temp_c: number;
    cells: Record<string, CellData>;
  }>;
}

export function App() {
  const [telemetry, setTelemetry] = useState<TelemetryPayload | null>(null);
  const [currentCycle, setCurrentCycle] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [packConfig, setPackConfig] = useState<string>('4S1P');
  const [selectedCell, setSelectedCell] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load default telemetry JSON
  useEffect(() => {
    fetch('/telemetry.json')
      .then((res) => {
        if (!res.ok) throw new Error('Default telemetry not found');
        return res.json();
      })
      .then((data: TelemetryPayload) => {
        setTelemetry(data);
        if (data.metadata?.pack_configuration) {
          setPackConfig(data.metadata.pack_configuration);
        }
        setIsLoading(false);
      })
      .catch(() => {
        // Generate built-in fallback simulation if file not placed in web/public
        const fallback = generateFallbackTelemetry(4, 50);
        setTelemetry(fallback);
        setIsLoading(false);
      });
  }, []);

  // Playback timer
  useEffect(() => {
    if (!isPlaying || !telemetry) return;
    const interval = setInterval(() => {
      setCurrentCycle((prev) => {
        if (prev >= telemetry.timeline.length) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 180 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, telemetry]);

  // Handle custom JSON upload from benchmark CLI
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (parsed.timeline && parsed.timeline.length > 0) {
          setTelemetry(parsed);
          setCurrentCycle(1);
          setIsPlaying(false);
          if (parsed.metadata?.pack_configuration) {
            setPackConfig(parsed.metadata.pack_configuration);
          }
        }
      } catch (err) {
        alert('Invalid benchmark telemetry JSON file');
      }
    };
    reader.readAsText(file);
  };

  // Switch pack size dynamically (4S to 12S)
  const handlePackChange = (newConfig: string) => {
    const sCount = parseInt(newConfig.replace(/[^0-9]/g, ''), 10) || 4;
    setPackConfig(newConfig);
    const newSim = generateFallbackTelemetry(sCount, 60);
    setTelemetry(newSim);
    setCurrentCycle(1);
    setIsPlaying(false);
    setSelectedCell(null);
  };

  if (isLoading || !telemetry) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950 text-slate-400">
        <div className="flex flex-col items-center space-y-4">
          <Activity className="w-8 h-8 text-emerald-500 animate-spin" />
          <p className="text-sm font-medium">Initializing 3D Battery Pack Digital Twin...</p>
        </div>
      </div>
    );
  }

  const activeStep = telemetry.timeline[currentCycle - 1] || telemetry.timeline[0];
  const maxCycles = telemetry.timeline.length;
  const packMeanSoh = (activeStep.pack_soh_mean * 100).toFixed(1);
  const packImbalance = (activeStep.pack_imbalance_delta * 100).toFixed(2);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Battery className="w-5 h-5 text-slate-950" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-lg text-slate-100 tracking-tight">PISTGAT-Battery-Bench</h1>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                v0.1.0 3D Twin
              </span>
            </div>
            <p className="text-xs text-slate-400">Physics-Informed Graph Attention SOH Benchmark</p>
          </div>
        </div>

        {/* Configuration switcher & File Upload */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center bg-slate-800/80 rounded-xl p-1 border border-slate-700/60 text-xs">
            {['4S1P', '6S1P', '8S1P', '12S1P'].map((cfg) => (
              <button
                key={cfg}
                onClick={() => handlePackChange(cfg)}
                className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                  packConfig.startsWith(cfg.slice(0, 2))
                    ? 'bg-emerald-500 text-slate-950 shadow-md font-bold'
                    : 'text-slate-300 hover:text-white'
                }`}
              >
                {cfg}
              </button>
            ))}
          </div>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".json"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center px-3.5 py-1.5 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs font-medium text-slate-200 shadow-sm transition-all"
          >
            <Upload className="w-3.5 h-3.5 mr-1.5 text-sky-400" />
            Upload JSON
          </button>

          <a
            href="https://github.com/pistgat-battery-bench/pistgat-battery-bench"
            target="_blank"
            rel="noreferrer"
            className="p-2 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition-all"
            title="GitHub Repository"
          >
            <Github className="w-4 h-4" />
          </a>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {/* Metric Cards HUD */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-400 block mb-1">Pack Mean SOH</span>
              <span className="text-2xl font-black text-emerald-400">{packMeanSoh}%</span>
            </div>
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-400 block mb-1">Pack Imbalance (ΔSOH)</span>
              <span className="text-2xl font-black text-amber-400">+{packImbalance}%</span>
            </div>
            <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
              <Activity className="w-5 h-5 text-amber-400" />
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-400 block mb-1">Max Cell Temp</span>
              <span className="text-2xl font-black text-slate-100">{activeStep.max_temp_c.toFixed(1)}°C</span>
            </div>
            <div className="w-10 h-10 rounded-xl bg-sky-500/10 flex items-center justify-center border border-sky-500/20">
              <Thermometer className="w-5 h-5 text-sky-400" />
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-lg flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-400 block mb-1">Cycle Step</span>
              <span className="text-2xl font-black text-slate-100">
                {currentCycle} <span className="text-xs text-slate-400 font-normal">/ {maxCycles}</span>
              </span>
            </div>
            <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center border border-purple-500/20">
              <Layers className="w-5 h-5 text-purple-400" />
            </div>
          </div>
        </div>

        {/* 3D Scene Section */}
        <div className="relative">
          <div className="h-[460px] w-full">
            <Canvas3D
              packConfig={packConfig}
              cells={activeStep.cells}
              selectedCell={selectedCell}
              onSelectCell={setSelectedCell}
            />
          </div>

          {/* Timeline Playback Overlay Controller */}
          <div className="mt-4 bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-10 h-10 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 flex items-center justify-center font-bold transition-all shadow-md shadow-emerald-500/20"
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
              </button>
              <button
                onClick={() => {
                  setCurrentCycle(1);
                  setIsPlaying(false);
                }}
                className="p-2.5 rounded-xl border border-slate-800 bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white transition-all"
                title="Reset to Cycle 1"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <div className="flex items-center bg-slate-950 rounded-xl border border-slate-800 p-1 text-xs">
                {[1, 2, 4].map((spd) => (
                  <button
                    key={spd}
                    onClick={() => setPlaybackSpeed(spd)}
                    className={`px-2 py-1 rounded-md font-mono ${
                      playbackSpeed === spd ? 'bg-slate-800 text-emerald-400 font-bold' : 'text-slate-400'
                    }`}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>

            {/* Slider */}
            <div className="flex-1 min-w-[200px] flex items-center space-x-3">
              <span className="text-xs font-mono text-slate-400">C1</span>
              <input
                type="range"
                min="1"
                max={maxCycles}
                value={currentCycle}
                onChange={(e) => {
                  setCurrentCycle(parseInt(e.target.value, 10));
                  setIsPlaying(false);
                }}
                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <span className="text-xs font-mono text-slate-400">C{maxCycles}</span>
            </div>

            <div className="text-xs font-mono text-slate-300">
              Degradation Progress: <span className="font-bold text-emerald-400">{((currentCycle / maxCycles) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* 2D Imbalance Heatmap Section */}
        <CellHeatmap
          cells={activeStep.cells}
          packImbalanceDelta={activeStep.pack_imbalance_delta}
          selectedCell={selectedCell}
          onSelectCell={setSelectedCell}
        />

        {/* Benchmark Curves & Scorecards */}
        <MetricsChart
          timeline={telemetry.timeline}
          benchmarkResults={telemetry.benchmark_results}
          currentCycle={currentCycle}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 text-slate-500 text-xs py-4 px-6 text-center">
        PISTGAT-Battery-Bench • Unified Lithium-Ion SOH Benchmark and 3D Pack Imbalance Visualizer • Open-Source MIT
      </footer>
    </div>
  );
}

// Fallback telemetry generator for instant standalone demonstration
function generateFallbackTelemetry(numCells: number, numCycles: number): TelemetryPayload {
  const timeline = [];
  const cellIds = Array.from({ length: numCells }, (_, i) => `Cell_${i + 1}`);

  for (let c = 1; c <= numCycles; c++) {
    const cellsObj: Record<string, CellData> = {};
    const sohVals: number[] = [];
    const tempVals: number[] = [];

    cellIds.forEach((cid, idx) => {
      const rate = 1.0 + idx * 0.12;
      const baseSoh = Math.max(0.68, 0.98 - (c / numCycles) * 0.28 * rate);
      const soh = baseSoh + (Math.sin(c * 0.4 + idx) * 0.005);
      const temp = 24.0 + 3.0 * idx + (1.0 - soh) * 12.0;
      const res = 0.075 + (1.0 - soh) * 0.08;

      cellsObj[cid] = {
        soh_pred: soh,
        soh_true: soh + (Math.random() - 0.5) * 0.008,
        temperature_c: temp,
        resistance_ohm: res,
        soh_deviation: 0
      };
      sohVals.push(soh);
      tempVals.push(temp);
    });

    const meanSoh = sohVals.reduce((a, b) => a + b, 0) / sohVals.length;
    const maxSoh = Math.max(...sohVals);
    const minSoh = Math.min(...sohVals);

    cellIds.forEach((cid) => {
      cellsObj[cid].soh_deviation = cellsObj[cid].soh_pred - meanSoh;
    });

    timeline.push({
      cycle: c,
      pack_soh_mean: meanSoh,
      pack_imbalance_delta: maxSoh - minSoh,
      max_temp_c: Math.max(...tempVals),
      cells: cellsObj
    });
  }

  return {
    metadata: {
      dataset: 'NASA B0005 Series Reference',
      pack_configuration: `${numCells}S1P`,
      nominal_capacity_ah: 2.0,
      cells_count: numCells
    },
    benchmark_results: {
      PISTGAT: { rmse_percent: 1.24, mae_percent: 1.05, r2_score: 0.9842, inference_latency_ms: 0.26 },
      LSTM: { rmse_percent: 2.15, mae_percent: 1.82, r2_score: 0.9521, inference_latency_ms: 0.48 },
      EKF: { rmse_percent: 3.42, mae_percent: 2.95, r2_score: 0.8914, inference_latency_ms: 0.08 }
    },
    timeline
  };
}
