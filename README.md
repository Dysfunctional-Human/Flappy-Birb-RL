# Flappy‑Birb‑RL 🐤

A lightweight **Flappy Bird** implementation in Pygame plus a tiny **neuro‑evolution** agent that learns to play—no deep‑learning frameworks required.

https://github.com/Dysfunctional-Human/Flappy-Birb-RL

---

## ✨ Features
- **Playable game loop** with smooth physics, sprite animation, and pixel‑perfect collisions.
- **Train & visualize** an evolutionary agent (small MLP) learning in real time.
- **No heavy deps**: just `pygame` and `numpy`.
- **Portable policy**: best agent weights saved as JSON for reuse.

---

## 🗂️ Project Structure
```
Flappy-Birb-RL/
├─ game.py                       # Play the game yourself
├─ agent_train_visualize.py      # Train & visualize the evolving agent
└─ components/
   ├─ agent.py                   # Minimal MLP policy (clone, mutate, act)
   ├─ bird.py                    # Bird physics + animation
   ├─ base.py                    # Scrolling ground
   ├─ pipes.py                   # Pipe generation + collisions
   └─ imgs/                      # Game art
      ├─ bg.png
      ├─ base.png
      ├─ pipe.png
      ├─ bird1.png
      ├─ bird2.png
      └─ bird3.png
```

---

## 🚀 Quickstart

### 1) Setup
```bash
# (Recommended) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install pygame numpy
```

### 2) Play the game
```bash
python game.py
```
Controls: **Space** to flap. Survive and pass pipes to score.

### 3) Train the agent (with live visualization)
```bash
python agent_train_visualize.py
```
- Watch up to 20 agents on screen while evolution runs generation‑by‑generation.
- The best policy for the current run is saved to **`best_agent_visual.json`**.
---

## 🧠 How the Agent Works
The agent is a small MLP (no backprop) optimized via **neuro‑evolution**.

### Observation Space
Each frame, the bird receives a 4‑dim vector:
```
[ dx, dy, vy, gap_half ]
```
- `dx`: horizontal distance to the next pipe’s leading edge
- `dy`: vertical distance from the bird’s center to the middle of the pipe gap
- `vy`: current vertical velocity of the bird
- `gap_half`: half the gap size between the upper and lower pipe

### Action Space
- **Binary**: `1` = jump, `0` = do nothing.
- The policy outputs a single scalar; jump if it’s **> 0**.

### Evolutionary Loop
- **Elitism**: top fraction cloned to next generation.
- **Mutation‑only** (no crossover): add Gaussian noise to weights/biases with per‑parameter probability.
- Fixed population & generations; RNG seed for reproducibility.

You can tweak hyperparameters (population size, elites, mutation sigma/prob, hidden sizes, generations, seed) directly in `agent_train_visualize.py`.

---

## 🎮 Game Mechanics (High‑level)
- **Physics**: quadratic fall with capped terminal step; upward tilt on jump, gradual downward tilt on fall.
- **Pipes**: random vertical position and gap; scroll at constant speed.
- **Collisions**: pixel‑perfect via Pygame masks.
- **HUD**: live generation stats (gen number, alive count, best fitness in run) during training.

## 🙌 Acknowledgements
- Pygame community & docs.
- Classic Flappy Bird for the inspiration 💛

