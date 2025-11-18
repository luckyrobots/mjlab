![Project banner](docs/static/mjlab-banner.jpg)

# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
</p>

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp) physics to provide lightweight, modular abstractions for RL robotics research and sim-to-real deployment.

> ⚠️ **BETA PREVIEW**
> mjlab is in active development. Expect **breaking changes** and **missing features** during the beta phase.
> There is **no stable release yet**. The PyPI package is only a snapshot — for the latest fixes and improvements, install from source or Git.

---

## Quick Start

mjlab requires an **NVIDIA GPU** for training (via MuJoCo Warp).
macOS is supported only for evaluation, which is significantly slower.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run the demo (no installation needed):

```bash
uvx --from mjlab --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9" demo
```

This launches an interactive viewer with a pre-trained Unitree G1 agent tracking a reference dance motion in MuJoCo Warp.

> ❓ Having issues? See the [FAQ](docs/faq.md).

**Try in Google Colab (no local setup required):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb)

Launch the demo directly in your browser with an interactive Viser viewer.

---

## Installation

**From source (recommended during beta):**

```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
uv run demo
```

**From PyPI (beta snapshot):**

```bash
uv add mjlab "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@486642c3fa262a989b482e0e506716d5793d61a9"
```

A Dockerfile is also provided.

For full setup instructions, see the [Installation Guide](docs/installation_guide.md).

---

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

---

### 2. Motion Imitation

Train a Unitree G1 to mimic reference motions. mjlab uses [WandB](https://wandb.ai) to manage reference motion datasets:

1. **Create a registry collection** in your WandB workspace named `Motions`

2. **Set your WandB entity**:
   ```bash
   export WANDB_ENTITY=your-organization-name
   ```

3. **Process and upload motion files**:
   ```bash
   MUJOCO_GL=egl uv run src/mjlab/scripts/csv_to_npz.py \
     --input-file /path/to/motion.csv \
     --output-name motion_name \
     --input-fps 30 \
     --output-fps 50 \
     --render  # Optional: generates preview video
   ```

> **Note**: For detailed motion preprocessing instructions, see the [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### Train and Play

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096

uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

---

### 3. Locomanipulation with OmniRetarget

Train a Unitree G1 on [OmniRetarget](https://omniretarget.github.io/) loco-manipulation motions.

1. **Download the OmniRetarget dataset:**

```bash
git lfs install
git clone https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset
```

2. **Convert URDF/SDF assets to MuJoCo MJCF (`.xml`) using `urdf2mjcf` (required):**

The OmniRetarget dataset ships URDF/SDF models under `OmniRetarget_Dataset/models/`.  
Running [`luckyrobots/urdf2mjcf`](https://github.com/luckyrobots/urdf2mjcf)'s batch converter in-place will generate matching `.xml` MJCF files next to each URDF/SDF; the locomanipulation task then uses `infer_object_cfg_from_motion_file` to automatically pick the correct MJCF asset based on each motion filename.

```bash
git clone https://github.com/luckyrobots/urdf2mjcf.git
cd urdf2mjcf
pip install -e .

# Convert all *.urdf files under the OmniRetarget models directory to *.xml.
./batch_convert_urdf.sh ../OmniRetarget_Dataset/models
```

3. **Convert OmniRetarget motions to mjlab format:**

This converts all `robot-object` trajectories to mjlab’s motion format; repeat for `robot-terrain/` and `robot-object-terrain/` if desired. Converting the full `robot-object` split typically takes **2–3 hours** end-to-end, depending on hardware.

```bash
mkdir -p artifacts/robot-object

for f in omniretarget/robot-object/*.npz; do
  uv run src/mjlab/scripts/omniretarget_to_mjlab.py \
    --input-file "$f" \
    --output-path artifacts/robot-object \
    --output-fps 50
done
```

#### Train and Play

```bash
MUJOCO_GL=egl uv run train Mjlab-Locomanipulation-Flat-Unitree-G1 \
  --motion-file artifacts/robot-object \
  --env.scene.num-envs 4096
```

`--motion-file` can point to a single converted `.npz` or a directory of motions; the environment will infer interactive objects/terrains from filenames and apply the locomanipulation rewards, terminations, and curriculum.


```bash
uv run play Mjlab-Locomanipulation-Flat-Unitree-G1 \
  --motion-file artifacts/robot-object \
  --checkpoint-file path/to/checkpoint.pt
```

When using `play`, `--motion-file` should match the converted OmniRetarget motions (file or directory), and `--checkpoint-file` should point to a checkpoint produced by the corresponding `train` run under `logs/rsl_rl/g1_locomanipulation/…`.

---

### 4. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP **before** training.

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions.
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions.
```

> [!NOTE]
> When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.

---

## Documentation

- **[Installation Guide](docs/installation_guide.md)**
- **[Why mjlab?](docs/motivation.md)**
- **[Migration Guide](docs/migration_guide.md)**
- **[FAQ & Troubleshooting](docs/faq.md)**

---

## Development

Run tests:

```bash
make test          # Run all tests
make test-fast     # Skip slow integration tests
```

Format code:

```bash
uvx pre-commit install
make format
```

---

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

The `third_party/` directory contains files from external projects, each with its own license:

- **isaaclab/** — [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) ([BSD-3-Clause](src/mjlab/third_party/isaaclab/LICENSE))

When distributing or modifying mjlab, comply with:
1. The Apache-2.0 license for mjlab’s original code
2. The respective licenses in `third_party/`

---

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for answering our questions, giving helpful feedback, and implementing features based on our requests countless times.
