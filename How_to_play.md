# How to play GoPet

## 1. Playing in the web UI

### Start the server

From the project root:

```bash
python run_play.py
```

By default the UI is at **http://127.0.0.1:5000/**.

Optional flags:

- `--port 5001` — use a different port
- `--debug` — Flask debug mode (auto-reloads on code changes)
- `--model path/to/model.pt` — expose an extra agent named `policy`

**First-time setup:** install the board renderer assets:

```bash
cd web
npm install
```

If jGoBoard is missing, the server prints a warning and the board will not render.

### Available agents

When the server starts, it loads agents from `agents/registry.py` whose checkpoints exist on disk (for example `basic_cnn`, `basic_cnn_5x5`). The `random` agent is always available.

For policy agents you can pick a **checkpoint** in the UI (for example `basic_cnn_epoch_010.pt`). Leave it on **default** to use the latest checkpoint (`*_latest.pt` when present).

### Human vs Bot

1. Open the URL in your browser.
2. Set **Mode** to **Human vs Bot**.
3. Choose **board size** (9, 13, or 19) and **your color** (Black or White).
4. Pick the **Bot agent** and optional **Bot checkpoint**.
5. Click intersections on the board to play. Use **Pass** when you want to pass.
6. The bot replies automatically when it is its turn. The move log shows bot moves and policy diagnostics (top moves, pass probability, score estimate when playing humans).

**New game** resets the board. **Undo** takes back one ply (human moves only; disabled while the bot is thinking).

### Bot vs Bot

1. Set **Mode** to **Bot vs Bot**.
2. Choose **Black agent** / **White agent** and their **checkpoints** independently. This lets you compare the same architecture at different training stages (for example epoch 1 vs epoch 10).
3. Click **New game** if needed. The two bots play automatically; you watch the game on the board and in the move log.
4. The game ends on two consecutive passes or a resignation.

Human color and pass controls are disabled in this mode.

---

## 2. Headless bot vs bot (win-rate evaluation)

Use `run_match.py` to simulate many games without the browser and print win rates. Colors alternate each game so neither side always has Black.

### Basic match

```bash
python run_match.py --agent-a basic_cnn --agent-b random --games 100 --board-size 9
```

### Same agent, different checkpoints

```bash
python run_match.py \
  --agent-a basic_cnn_5x5 \
  --agent-b basic_cnn_5x5 \
  --checkpoint-a agents/basic_cnn_5x5/checkpoints/basic_cnn_5x5_epoch_001.pt \
  --checkpoint-b agents/basic_cnn_5x5/checkpoints/basic_cnn_5x5_epoch_010.pt \
  --games 200 \
  --board-size 19
```

### Useful options

| Flag | Default | Meaning |
|------|---------|---------|
| `--games` | 1000 | Number of games to play |
| `--board-size` | 19 | Board size |
| `--komi` | 7.5 | Komi (applied only when the game ends by scoring) |
| `--max-moves` | `2 × board_size²` | Force-stop and score unfinished games after this many plies |
| `--checkpoint-a` / `--checkpoint-b` | default checkpoint | Override model file for each side |
| `--seed` | 0 | RNG seed for the `random` agent |
| `--quiet` | off | Hide progress output |

### Output

The script prints a summary: wins per agent, win rates, Black/White wins, resignations, max-move stops, and end reasons. Use this to compare agents or checkpoints quantitatively.

**Note:** Agents must be trained first; if a checkpoint is missing, the command fails with a clear error. List available agent ids via the choices shown in `python run_match.py --help`.
