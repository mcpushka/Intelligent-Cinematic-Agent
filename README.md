# Intelligent Cinematic Agent

**Autonomous Video Generation from 3D Gaussian Splatting Scenes**

An AI-powered cinematic agent that autonomously explores 3D Gaussian Splatting scenes, plans smooth camera trajectories with obstacle avoidance, and generates professional panorama tour videos.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Description](#algorithm-description)
- [Dependencies](#dependencies)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for large scenes)
- CUDA Toolkit 11.7+

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Intelligent-Cinematic-Agent.git
   cd Intelligent-Cinematic-Agent
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA installation:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## Usage

### Basic Usage

Generate a panorama tour video from a single scene:

```bash
python -m src.main \
  --scenes path/to/scene.ply \
  --output-dir outputs \
  --duration 60 \
  --fps 24 \
  --resolution 1280x720 \
  --device cuda
```

### Multiple Scenes

Process multiple scenes in one run:

```bash
python -m src.main \
  --scenes ConferenceHall.ply Museum.ply outdoor-drone.ply outdoor-street.ply Theater.ply \
  --output-dir outputs \
  --duration 60 \
  --fps 24 \
  --resolution 1280x720 \
  --device cuda
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--scenes` | List of .ply scene files (required) | - |
| `--output-dir` | Directory for output videos | `outputs` |
| `--duration` | Video duration in seconds | `60.0` |
| `--fps` | Frames per second | `24` |
| `--resolution` | Video resolution (WxH) | `1280x720` |
| `--device` | Compute device (`cuda` or `cpu`) | Auto-detect |
| `--backend` | Rendering backend | `gsplat` |
| `--radius-clip` | Radius clipping for gsplat | `0.0` |

### Output Structure

```
outputs/
├── ConferenceHall/
│   └── panorama_tour.mp4
├── Museum/
│   └── panorama_tour.mp4
└── ...
```

### Example: Quick Test (Low Memory)

For GPUs with limited memory (e.g., 16GB):

```bash
python -m src.main \
  --scenes scene.ply \
  --output-dir outputs \
  --duration 30 \
  --fps 15 \
  --resolution 960x540 \
  --device cuda
```

---

## Algorithm Description

### 1. Scene Loading (`gaussians.py`)

The system supports multiple PLY formats:

- **SuperSplat Packed Format**: Efficiently unpacks compressed Gaussian data using chunk metadata for position, rotation, scale, and color denormalization.
- **Standard Gaussian Splatting Format**: Loads vertex data with explicit x, y, z, scale, rotation, opacity, and color fields.
- **Chunk-only Format**: Uses chunk centers as Gaussian positions when packed vertex data is unavailable.

**Key Features:**
- Automatic format detection
- Downsampling for memory efficiency (configurable max_gaussians)
- Color normalization (0-255 → 0-1)
- Quaternion normalization for stable rendering

### 2. Path Planning (`path_planner.py`)

The path planning system generates smooth camera trajectories inside the scene:

#### 2.1 Occupancy Grid Construction
- Builds a coarse 3D voxel grid from Gaussian positions
- Marks cells containing Gaussians as occupied
- Grid resolution: 32×32×32 (configurable)

#### 2.2 A* Pathfinding
- Finds collision-free paths between random start/goal positions
- Uses Manhattan distance heuristic
- 6-connectivity (cardinal directions only)
- Fallback to straight-line interpolation if no path found

#### 2.3 Path Interpolation
- Linear interpolation along the discrete grid path
- Generates smooth camera positions for each frame
- Camera always looks toward scene center (guarantees visibility)

#### 2.4 View Matrix Generation
- Constructs world-to-camera matrices using look-at convention
- Up vector: (0, 1, 0)
- Ensures camera stays within scene bounding box

### 3. Scene Exploration (`explorer.py`)

The `SceneExplorer` class orchestrates the exploration:

- Optional scene normalization for consistent behavior
- Configurable camera height
- Reproducible paths via random seed

### 4. Rendering (`renderer.py`)

Uses `gsplat` library for efficient Gaussian splatting:

- Batched rendering for performance
- Pinhole camera model with configurable FOV
- RGB output with alpha channel
- Direct MP4 export via imageio

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Deep learning framework, CUDA support |
| `gsplat` | ≥0.1.0 | Gaussian splatting rasterization |
| `numpy` | ≥1.20 | Numerical operations |
| `plyfile` | ≥0.8 | PLY file parsing |
| `imageio` | ≥2.9 | Video encoding |
| `imageio-ffmpeg` | ≥0.4 | FFmpeg backend for MP4 |
| `tqdm` | ≥4.60 | Progress bars |

### Installation via requirements.txt

```bash
pip install -r requirements.txt
```

### requirements.txt Contents

```
torch>=2.1
numpy
plyfile
tqdm
imageio
imageio-ffmpeg
gsplat
```

---

## Known Limitations

### Memory Constraints

1. **Large Scenes**: Scenes with millions of Gaussians may exceed GPU memory. The system automatically downsamples to 200K Gaussians by default.

2. **Batch Size**: Reduced to 1 frame per batch for memory safety. This increases rendering time but prevents OOM errors.

3. **Resolution**: Higher resolutions (e.g., 4K) require significantly more memory. Recommended: 1280×720 for 16GB GPUs.

### Path Planning

1. **Grid Resolution**: The 32³ occupancy grid may miss small obstacles or create overly conservative paths in complex scenes.

2. **Dense Scenes**: If the scene is extremely dense (>90% occupied), the system falls back to orbital camera motion around the center.

3. **Single Path**: Currently generates one path per scene. Multiple interesting viewpoints are not automatically detected.

### Rendering

1. **First Run Compilation**: gsplat compiles CUDA kernels on first use, which can take 5-10 minutes.

2. **Color Accuracy**: Packed format color unpacking may have slight quantization artifacts.

3. **No View-Dependent Effects**: Current implementation uses RGB colors only, not full spherical harmonics.

### Format Support

1. **PLY Only**: Only .ply files are supported. Other formats (e.g., .splat) require conversion.

2. **Packed Format**: Assumes specific bit-packing scheme (SuperSplat). Other packed formats may not work.

### Performance

| Scene Size | GPU Memory | Approx. Render Time (60s video) |
|------------|------------|--------------------------------|
| 100K Gaussians | ~4GB | ~5 min |
| 200K Gaussians | ~8GB | ~10 min |
| 500K Gaussians | ~14GB | ~20 min |

---

## Project Structure

```
Intelligent-Cinematic-Agent/
├── src/
│   ├── main.py           # Main entry point and CLI
│   ├── gaussians.py      # Scene loading and Gaussian data handling
│   ├── explorer.py       # Scene exploration agent
│   ├── path_planner.py   # A* pathfinding and trajectory generation
│   └── renderer.py       # gsplat-based video rendering
├── outputs/              # Generated videos (created automatically)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── task.md              # Assignment specification
```

---
