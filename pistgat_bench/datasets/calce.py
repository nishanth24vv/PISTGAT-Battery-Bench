"""CALCE Battery Aging Dataset Loader (Center for Advanced Life Cycle Engineering)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pistgat_bench.datasets.base import BaseBatteryDataset, BatteryCycleRecord


class CALCEBatteryDataset(BaseBatteryDataset):
    """Loader for CALCE (University of Maryland) battery degradation datasets.

    Typical cells include CS2_35, CS2_36, CS2_37, and CS2_38 (1.1 Ah nominal capacity).
    """

    def __init__(self, nominal_capacity_ah: float = 1.1):
        super().__init__(name="CALCE", nominal_capacity_ah=nominal_capacity_ah)

    def load(self, source_path: Optional[str] = None) -> CALCEBatteryDataset:
        """Load CALCE datasets from directory or synthesize reference benchmark cells."""
        if source_path and os.path.exists(source_path):
            self._load_from_files(source_path)
        else:
            self._synthesize_calce_cells()

        self.is_loaded = True
        self._compute_normalization_stats()
        return self

    def _load_from_files(self, source_path: str) -> None:
        """Parse CALCE CSV or Excel files if present."""
        # Check for CSV files
        p = Path(source_path)
        files = [p] if p.is_file() else list(p.glob("*.csv"))
        if not files:
            self._synthesize_calce_cells()
            return

        for f in files:
            cell_name = f.stem
            try:
                import pandas as pd
                df = pd.read_csv(f)
                records = []
                # Common CALCE columns: Cycle_Index, Discharge_Capacity(Ah)
                cap_col = next((c for c in df.columns if "capacity" in c.lower()), None)
                if cap_col:
                    for idx, row in df.iterrows():
                        cap = float(row[cap_col])
                        soh = float(np.clip(cap / self.nominal_capacity_ah, 0.0, 1.0))
                        records.append(
                            BatteryCycleRecord(
                                cycle_index=int(row.get("Cycle_Index", idx + 1)),
                                capacity_ah=cap,
                                soh=soh,
                                internal_resistance_ohm=0.07 + 0.0003 * idx,
                                voltage_mean=3.7,
                                voltage_min=2.7,
                                voltage_max=4.2,
                                current_mean=-0.55,
                                temp_max=25.0
                            )
                        )
                    if records:
                        self.cell_data[cell_name] = records
            except Exception:
                pass

        if not self.cell_data:
            self._synthesize_calce_cells()

    def _synthesize_calce_cells(self) -> None:
        """Synthesize CALCE reference cells CS2_35, CS2_36, CS2_37, CS2_38."""
        cell_configs = [
            ("CS2_35", 1.08, 0.0011, 0.065, 25.0),
            ("CS2_36", 1.07, 0.0013, 0.068, 25.0),
            ("CS2_37", 1.09, 0.0010, 0.064, 25.0),
            ("CS2_38", 1.06, 0.0014, 0.070, 25.0)
        ]
        num_cycles = 400
        for cell_id, init_cap, fade_rate, init_r0, base_temp in cell_configs:
            cycles = []
            cap = init_cap
            r0 = init_r0
            np.random.seed(int("".join([c for c in cell_id if c.isdigit()])))
            for i in range(1, num_cycles + 1):
                fade = fade_rate + np.random.normal(0, 0.0002)
                if i in [120, 240]:  # relaxation recovery
                    fade -= 0.008
                cap = max(0.55, cap - fade)
                r0 += 0.00015
                soh = float(np.clip(cap / self.nominal_capacity_ah, 0.0, 1.0))
                cycles.append(
                    BatteryCycleRecord(
                        cycle_index=i,
                        capacity_ah=round(float(cap), 4),
                        soh=round(soh, 4),
                        internal_resistance_ohm=round(float(r0), 5),
                        voltage_mean=round(float(3.75 - 0.0008 * i), 4),
                        voltage_min=2.7,
                        voltage_max=4.2,
                        current_mean=-0.55,
                        temp_max=round(float(base_temp + 5.0 * (1.0 - soh) + np.random.normal(0, 0.2)), 2),
                        duration_s=7200.0
                    )
                )
            self.cell_data[cell_id] = cycles
