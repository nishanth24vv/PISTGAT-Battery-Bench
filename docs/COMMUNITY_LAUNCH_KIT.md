# PISTGAT-Battery-Bench Community Distribution & Launch Kit

Copy-pasteable announcement templates tailored for each developer platform and community.

---

## 1. Reddit (Project Demonstrations)
**Target Subreddits**: `r/MachineLearning`, `r/batteries`, `r/Python`, `r/electricvehicles`, `r/computervision`

### Post Title
> **[P] I built an open-source Li-ion Battery SOH Benchmark with a customizable 3D Pack Digital Twin in Three.js (PISTGAT-Battery-Bench)**

### Post Body
```markdown
Hey everyone!

One of the biggest headaches in Battery Management Systems (BMS) and EV research is dealing with fragmented battery degradation datasets (NASA, CALCE, Oxford all use completely different file formats, sampling rates, and schemas) and reproducible baseline models.

To solve this, I built and open-sourced **PISTGAT-Battery-Bench**: a unified benchmarking suite and interactive 3D pack visualizer.

🔗 **GitHub**: https://github.com/nishanth24vv/PISTGAT-Battery-Bench  
🎮 **Live 3D Web Visualizer**: https://pistgat-bench.vercel.app  

### What's inside:
1. **Unified Dataset Ingestion**: Ingests NASA (B0005, B0006..), CALCE (CS2..), and Oxford pouch cell series with standard normalization, sliding windows, and Leave-One-Cell-Out (LOCO) splits.
2. **Physics-Informed Spatiotemporal Model (PISTGAT)**:
   - **Spatial Graph Attention (GATv2)**: Models thermal and electrical coupling across cell strings (4S to 12S+).
   - **Temporal Transformer**: Captures long-horizon degradation fade without vanishing gradients.
   - **Capacity Monotonicity Loss**: Penalizes non-physical capacity recoveries during continuous cycling.
3. **Automotive Baselines**: Side-by-side benchmarking against Extended Kalman Filters (1RC Thevenin ECM) and Bidirectional LSTMs.
4. **Customizable 3D Digital Twin (React + Three.js)**:
   - You can completely adapt the 3D model, cell arrangement (4S1P to 12S1P), and visual shaders using your own personal UI choices.
   - Drag-and-drop custom telemetry JSON exported directly from the CLI.
5. **Keyless Local Execution**:
   ```bash
   pip install -e .
   pistgat-bench run --dataset nasa --sample
   ```

*(Attach `docs/demo.gif` to the Reddit post as a video/image demonstration)*

Would love any feedback on the graph attention formulation or additional battery datasets you'd like supported!
```

---

## 2. LinkedIn (Project Development Journey)

### Post Body
```markdown
🔋 Excited to open-source PISTGAT-Battery-Bench: A unified Lithium-Ion State of Health (SOH) benchmark paired with a customizable 3D battery pack digital twin.

In the Electric Vehicle and Battery Management System (BMS) domain, predicting cell degradation and pack-level imbalance is critical for safety and second-life diagnostics. However, researchers often spend 70% of their time wrestling with incompatible raw data files across NASA, CALCE, and Oxford repositories.

I developed this framework to bridge battery physics with modern deep learning and interactive web graphics:

💡 **Key Engineering Highlights:**
• **Spatiotemporal Graph Attention (GATv2) + Transformer**: Modeled cell-to-cell thermal/electrical coupling across 4S–12S strings, pairing it with self-attention across historical cycle windows.
• **Physics-Informed Monotonicity Loss**: Standard neural networks often predict unphysical capacity recovery. Formulating a custom monotonicity constraint reduced non-physical trajectory oscillations.
• **Standardized Benchmarking**: Evaluated against traditional automotive baselines (Extended Kalman Filter with 1RC ECM) and Bidirectional LSTMs, logging real RMSE, MAE, R², and latency metrics to MLflow.
• **Customizable Three.js 3D Pack Visualizer**: Built a decoupled, modular 3D digital twin in React & Three.js that users can easily style and tailor to their own UI preferences and pack geometries.

🔗 **GitHub**: https://github.com/nishanth24vv/PISTGAT-Battery-Bench
🌐 **Live Demo**: https://pistgat-bench.vercel.app

Check out the 15-second demo below! I would love your thoughts and feedback.

#MachineLearning #BatteryManagement #ElectricVehicles #PyTorch #ThreeJS #OpenSource #DeepLearning #DataScience
```

---

## 3. Dev.to (Technical Article)

### Article Title
> **Benchmarking Li-Ion Battery State of Health: Graph Attention Networks, Physics Losses, and 3D Web Twins**

### Article Tags
`#python` `#datascience` `#machinelearning` `#webdev`

### Markdown Content
```markdown
Battery State of Health (SOH) estimation is one of the most critical challenges in electric mobility and grid storage. Yet, benchmarking machine learning models against traditional control filters (like the Extended Kalman Filter) is often hindered by disparate data schemas.

In this article, we break down the architecture of **PISTGAT-Battery-Bench** — an open-source framework combining PyTorch Geometric, temporal Transformers, physics-informed losses, and a customizable Three.js 3D digital twin.

---

### The Problem: Why Standard LSTMs Fail on Battery Packs
1. **Inter-cell coupling**: Cells in a series string do not age in isolation; thermal proximity and current distribution cause localized degradation hotspots.
2. **Unphysical predictions**: Purely data-driven neural nets can predict sudden 5% capacity jumps in the middle of active cycling.
3. **Fragmented formats**: NASA Ames uses `.mat` structures, CALCE uses heterogeneous Excel tables, and Oxford uses dynamic drive profiles.

---

### 1. Spatiotemporal Graph Attention (GATv2)
Rather than treating a multi-cell pack as independent time-series, we represent the pack as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where each node represents a cell:

```python
from torch_geometric.nn import GATv2Conv

# Spatial attention over adjacent series/thermal neighbors
h_spatial = F.elu(self.gat1(node_features, edge_index))
h_spatial = self.gat2(h_spatial, edge_index)
```

---

### 2. Physics-Informed Capacity Monotonicity Loss
To prevent non-physical capacity increases during degradation, we penalize positive increments:

$$\mathcal{L}_{\text{mono}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \max\left(0, \hat{y}_{t+1} - \hat{y}_t - \epsilon_{\text{regen}}\right)^2$$

```python
class CapacityMonotonicityLoss(nn.Module):
    def __init__(self, epsilon_regen=0.005, penalty_weight=1.0):
        super().__init__()
        self.epsilon_regen = epsilon_regen
        self.penalty_weight = penalty_weight

    def forward(self, y_pred):
        diff = y_pred[1:] - y_pred[:-1]
        violations = torch.relu(diff - self.epsilon_regen)
        return self.penalty_weight * torch.mean(torch.square(violations))
```

---

### 3. Customizable 3D Pack Digital Twin
The Python CLI exports standardized telemetry JSON:
```bash
pistgat-bench run --dataset nasa --sample --export-telemetry web/public/telemetry.json
```

The frontend uses React and Three.js to render 4S to 12S pack geometries with dynamic SOH heatmaps. Because the visualizer is built from modular components, developers can easily customize the shaders, materials, and UI layout to match their own dashboard design.

Explore the code on GitHub: [nishanth24vv/PISTGAT-Battery-Bench](https://github.com/nishanth24vv/PISTGAT-Battery-Bench).
```

---

## 4. Hashnode (Tutorial)

### Title
> **How to Build a Battery Pack Digital Twin with PyTorch Geometric and Three.js**

### Subtitle
> A complete step-by-step tutorial on ingesting battery aging datasets, training spatiotemporal GNNs, and rendering 3D cell imbalance heatmaps.

### Key Sections Outline
1. **Introduction**: Why Lithium-Ion digital twins require spatiotemporal modeling.
2. **Setting Up the Unified Data Ingestion**: Abstracting NASA, CALCE, and Oxford datasets.
3. **Implementing the Baseline EKF vs. PISTGAT**:
   - Discrete 1RC Thevenin ECM state equations.
   - GATv2 + Temporal Transformer architecture.
4. **Enforcing Physical Constraints**: Why monotonicity loss is mandatory for battery aging.
5. **Connecting to Three.js**:
   - Building 18650 cylindrical geometry.
   - Dynamic vertex coloring based on real-time cell SOH.
   - Orbit controls, timeline scrubbing, and custom UI integration.
6. **Deploying on Vercel**: Zero-config deployment with `vercel.json`.

*(Full code and interactive repository link included)*

---

## 5. Hacker News (Show HN)

### Title
> **Show HN: PISTGAT-Battery-Bench – Unified Li-ion SOH benchmark and customizable 3D pack twin**

### Submission URL
`https://github.com/nishanth24vv/PISTGAT-Battery-Bench`

### First Comment (Text)
```text
Hi HN!

I built PISTGAT-Battery-Bench, an open-source benchmarking suite and interactive 3D digital twin for Lithium-Ion Battery Management Systems (BMS).

Repo: https://github.com/nishanth24vv/PISTGAT-Battery-Bench
Live 3D Visualizer: https://pistgat-bench.vercel.app

### Why I built this:
Battery degradation datasets (NASA Prognostics, CALCE, Oxford) are notoriously inconsistent in their file formats (.mat, CSV, Excel), sampling rates, and evaluation procedures. Furthermore, comparing modern neural network models against classic automotive filters (Extended Kalman Filter with Thevenin Equivalent Circuit Models) usually requires writing custom evaluation pipelines from scratch.

### What it does:
- Unified Data Loading: Standard interface for NASA B0005 series, CALCE CS2, and Oxford cells with Leave-One-Cell-Out (LOCO) cross-validation.
- PISTGAT Model: Combines Spatial Graph Attention Networks (GATv2) to capture inter-cell electrical and thermal coupling across 4S-12S packs with a temporal Transformer encoder and a physics-informed capacity monotonicity loss.
- Customizable 3D Visualizer: A React and Three.js frontend where you can adjust pack configurations (4S-12S+), customize the UI styling, and drag-and-drop exported benchmark JSON telemetry.
- Zero-Credential CLI: Runs immediately offline via `pip install -e .` and `pistgat-bench run --dataset nasa --sample`.

I’d love to hear your thoughts, especially from anyone working in EV battery diagnostics or spatiotemporal graph networks!
```

---

## 6. Discord Communities

### Target Discords
- **PyTorch Discord** (`#machine-learning-projects`, `#showcase`)
- **Three.js Discord** (`#showcase`)
- **Open-Source Battery Data / Battery Tech Discords**
- **MLflow / MLOps Community**

### Message Snippet
```text
Hey everyone! 👋 Just published an open-source project combining PyTorch Geometric, Transformers, and Three.js:

🔋 **PISTGAT-Battery-Bench**: A unified Li-ion battery State of Health benchmark and customizable 3D pack digital twin.

✨ Highlights:
- Standardized loader for NASA, CALCE, and Oxford battery degradation datasets.
- Spatiotemporal GAT + Transformer model with physics-guided capacity monotonicity loss.
- Side-by-side benchmarking against automotive EKF (1RC ECM) and LSTM baselines with MLflow.
- Customizable 3D pack visualizer in Three.js (4S to 12S series strings) with dynamic cell imbalance heatmaps and custom UI adaptation.

⭐ GitHub: https://github.com/nishanth24vv/PISTGAT-Battery-Bench
🎮 Live Demo: https://pistgat-bench.vercel.app

Feedback and PRs are super welcome!
```
