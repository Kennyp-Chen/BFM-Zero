# Repository Guidelines

## Project Structure & Module Organization

- `humanoidverse/` contains the training, inference, simulator, agent, utility, and motion-library Python packages.
- `humanoidverse/config/` stores Hydra YAML configuration grouped by experiment, robot, simulator, rewards, observations, and callbacks.
- `data_process/` contains motion-conversion and dataset-generation scripts; `dataset/` stores local motion data.
- `static/`, `model/`, `huiying/`, and `logs/` contain visual assets, checkpoints, experiment outputs, and run records. Treat generated artifacts as data, not source.
- `docs/` contains checkpoint/playback notes. Add new reusable documentation there rather than embedding long procedures in code.

## Build, Test, and Development Commands

Create the documented environment with `uv sync` (or use the project Conda environment), then run commands from the repository root.

```bash
uv run python -m humanoidverse.train                 # start training
uv run python -m humanoidverse.tracking_inference --help
uv run python -m humanoidverse.visualize_motion --help
uv run ruff check humanoidverse data_process              # lint
```

Isaac Sim requires its supported Linux installation and GPU; use `--simulator mujoco` for MuJoCo-only inference or visualization where supported. Large motion/checkpoint files may require `git lfs pull`.

## Coding Style & Naming Conventions

Use Python 3.10/3.11-compatible code, four-space indentation, descriptive `snake_case` names, and `PascalCase` for classes. Keep imports explicit and functions focused. Ruff is configured with a 140-character line limit and import sorting; run it before submitting changes. Match existing Hydra naming conventions for YAML files and robot/config identifiers.

## Testing Guidelines

There is no dedicated automated test suite currently checked in. For changes, run `uv run ruff check ...` and perform the narrowest relevant smoke test: import the changed module, invoke its `--help`, or run a short headless MuJoCo rollout. Do not require Isaac Sim for tests that can exercise MuJoCo or pure Python paths.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase summaries such as `fix bug of piplus action range` and `Add new robot ...`. Keep commits focused and explain the behavioral impact. Pull requests should include a concise description, affected robot/simulator/config, validation commands and results, linked issue when applicable, and screenshots or videos for visualization or policy-behavior changes.

## Configuration & Data Safety

Do not commit secrets, private machine paths, or regenerated checkpoints/logs unless explicitly required. Preserve existing user changes and keep robot asset, motion-data, and config updates synchronized so names and paths resolve in both Isaac Sim and MuJoCo.
