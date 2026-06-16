import {
  MARK,
  STONE,
  createGame,
  createRenderer,
  rules,
  setAssetBaseUrl,
} from "/static/vendor/jgoboard/dist/jgoboard.js";

const ASSET_BASE = "/static/vendor/jgoboard/";
setAssetBaseUrl(ASSET_BASE);

const ui = {
  boardSize: document.getElementById("board-size"),
  theme: document.getElementById("theme"),
  agent: document.getElementById("agent"),
  humanColor: document.getElementById("human-color"),
  toPlay: document.getElementById("to-play"),
  moveNumber: document.getElementById("move-number"),
  blackCaptures: document.getElementById("black-captures"),
  whiteCaptures: document.getElementById("white-captures"),
  koPoint: document.getElementById("ko-point"),
  status: document.getElementById("status"),
  spinner: document.getElementById("spinner"),
  log: document.getElementById("log"),
  passBtn: document.getElementById("pass"),
  resetBtn: document.getElementById("reset"),
  undoBtn: document.getElementById("undo"),
};

const labels = {
  [STONE.BLACK]: "Black",
  [STONE.WHITE]: "White",
};

let game = null;
let renderer = null;
let moveRecord = [];
let waitingForBot = false;
let gameOver = false;

function addLog(text) {
  const line = document.createElement("div");
  line.textContent = text;
  ui.log.appendChild(line);
  ui.log.scrollTop = ui.log.scrollHeight;
}

function formatPct(prob) {
  return `${(prob * 100).toFixed(1)}%`;
}

function formatScoreEstimate(scoreEstimate) {
  if (!scoreEstimate || scoreEstimate.disabled) {
    return null;
  }
  if (scoreEstimate.error) {
    return `score error: ${scoreEstimate.error}`;
  }
  if (!scoreEstimate.estimate) {
    return null;
  }
  const margin = scoreEstimate.bot_margin;
  if (typeof margin === "number") {
    const ahead = margin >= 0 ? "bot ahead" : "bot behind";
    return `score ${scoreEstimate.estimate} (${ahead} ${Math.abs(margin).toFixed(1)})`;
  }
  return `score ${scoreEstimate.estimate}`;
}

function formatDiagnostics(diagnostics) {
  const parts = [];

  if (diagnostics.top_moves && diagnostics.top_moves.length > 0) {
    const ranked = diagnostics.top_moves
      .map((entry) => `#${entry.rank} ${entry.move} ${formatPct(entry.prob)}`)
      .join(", ");
    parts.push(ranked);
    parts.push(`pass ${formatPct(diagnostics.pass_prob ?? 0)}`);
  }

  const scoreText = formatScoreEstimate(diagnostics.score_estimate);
  if (scoreText) {
    parts.push(scoreText);
  }

  if (parts.length === 0) {
    return `diagnostics: ${JSON.stringify(diagnostics)}`;
  }

  return `diagnostics: ${parts.join(", ")}`;
}

function setStatus(text) {
  ui.status.textContent = text;
}

function setWaiting(active) {
  waitingForBot = active;
  ui.spinner.classList.toggle("active", active);
}

function humanPlayerStone() {
  return ui.humanColor.value === "white" ? STONE.WHITE : STONE.BLACK;
}

function botPlayerStone() {
  return humanPlayerStone() === STONE.BLACK ? STONE.WHITE : STONE.BLACK;
}

function isHumanTurn() {
  return !gameOver && !waitingForBot && game.currentPlayer === humanPlayerStone();
}

function refreshMarkers() {
  game.board.each((point, intersection) => {
    if (intersection.mark !== MARK.NONE) {
      game.board.setMark(point, MARK.NONE);
    }
  });

  if (game.lastMove) {
    game.board.setMark(game.lastMove, MARK.CIRCLE);
  }
  if (game.koPoint) {
    game.board.setMark(game.koPoint, MARK.SQUARE);
  }
}

function updateUI() {
  const state = game.getState();
  ui.toPlay.textContent = labels[state.currentPlayer];
  ui.moveNumber.textContent = String(state.moveNumber);
  ui.blackCaptures.textContent = String(state.captures.black);
  ui.whiteCaptures.textContent = String(state.captures.white);
  ui.koPoint.textContent = state.ko || "none";
  ui.undoBtn.disabled = !state.canUndo || waitingForBot;

  if (gameOver) {
    setStatus("Game over");
  } else if (waitingForBot) {
    setStatus("Waiting for bot...");
  } else if (isHumanTurn()) {
    setStatus(`Your turn (${labels[humanPlayerStone()]})`);
  } else {
    setStatus(`Bot thinking (${labels[botPlayerStone()]})`);
  }
}

function checkGameOver() {
  const n = moveRecord.length;
  if (n >= 2 && moveRecord[n - 1] === "pass" && moveRecord[n - 2] === "pass") {
    gameOver = true;
    addLog("Game ended (two passes)");
    return true;
  }
  if (n >= 1 && moveRecord[n - 1] === "resign") {
    gameOver = true;
    return true;
  }
  return false;
}

async function requestBotMove() {
  if (gameOver || game.currentPlayer !== botPlayerStone()) {
    return;
  }

  setWaiting(true);
  updateUI();

  try {
    const response = await fetch(`/api/select-move/${ui.agent.value}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        board_size: parseInt(ui.boardSize.value, 10),
        moves: moveRecord.slice(),
        against_human: true,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Bot request failed");
    }

    if (!waitingForBot) {
      return;
    }

    applyServerMove(data.bot_move, "Bot");
    if (data.diagnostics && Object.keys(data.diagnostics).length > 0) {
      addLog(formatDiagnostics(data.diagnostics));
    }
  } catch (error) {
    addLog(`bot error: ${error.message}`);
    setStatus(error.message);
  } finally {
    setWaiting(false);
    refreshMarkers();
    updateUI();
    maybeTriggerBot();
  }
}

function applyServerMove(vertex, prefix) {
  if (vertex === "pass") {
    game.pass();
    moveRecord.push("pass");
    addLog(`${prefix} pass`);
  } else if (vertex === "resign") {
    moveRecord.push("resign");
    gameOver = true;
    addLog(`${prefix} resigns`);
    return;
  } else {
    const result = game.play(vertex);
    if (!result.ok) {
      addLog(`${prefix} illegal ${vertex}: ${result.code}`);
      return;
    }
    moveRecord.push(vertex);
    const captureInfo = result.captures.length ? ` x${result.captures.length}` : "";
    addLog(`${prefix} ${vertex}${captureInfo}`);
  }

  checkGameOver();
}

function handleHumanPlay(vertex) {
  if (!isHumanTurn()) {
    return;
  }

  const result = game.play(vertex);
  if (!result.ok) {
    addLog(`illegal ${labels[game.currentPlayer]} ${vertex}: ${result.code}`);
    return;
  }

  moveRecord.push(vertex);
  const captureInfo = result.captures.length ? ` x${result.captures.length}` : "";
  addLog(`You ${vertex}${captureInfo}`);
  refreshMarkers();
  updateUI();

  if (checkGameOver()) {
    updateUI();
    return;
  }

  maybeTriggerBot();
}

function maybeTriggerBot() {
  if (!gameOver && game.currentPlayer === botPlayerStone() && !waitingForBot) {
    requestBotMove();
  }
}

async function createBoardSession() {
  const size = parseInt(ui.boardSize.value, 10);

  if (renderer) {
    renderer.destroy();
  }

  game = createGame({
    size,
    rules: rules.japanese({ ko: "simple", suicide: "forbidden" }),
  });

  renderer = createRenderer("#board", {
    board: game.board,
    theme: ui.theme.value,
    assetBaseUrl: ASSET_BASE,
  });

  renderer.enableHoverPreview({
    stone: () => (isHumanTurn() ? game.currentPlayer : null),
  });

  renderer.on("click", ({ vertex }) => {
    if (!vertex) {
      return;
    }
    handleHumanPlay(vertex);
  });

  await renderer.whenReady();
  renderer.render();
}

async function resetGame() {
  moveRecord = [];
  gameOver = false;
  waitingForBot = false;
  ui.log.textContent = "";
  await createBoardSession();
  refreshMarkers();
  addLog("New game");
  updateUI();
  maybeTriggerBot();
}

async function init() {
  try {
    const response = await fetch("/api/agents");
    const data = await response.json();
    ui.agent.innerHTML = "";
    for (const name of data.agents) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      ui.agent.appendChild(option);
    }
  } catch (error) {
    ui.agent.innerHTML = '<option value="random">random</option>';
    addLog(`Could not load agents: ${error.message}`);
  }

  ui.passBtn.addEventListener("click", () => {
    if (!isHumanTurn()) {
      return;
    }
    game.pass();
    moveRecord.push("pass");
    addLog("You pass");
    refreshMarkers();
    updateUI();
    if (!checkGameOver()) {
      maybeTriggerBot();
    }
  });

  ui.resetBtn.addEventListener("click", () => {
    resetGame();
  });

  ui.undoBtn.addEventListener("click", () => {
    if (waitingForBot || !game.getState().canUndo) {
      return;
    }
    game.undo();
    moveRecord.pop();
    gameOver = false;
    addLog("Undo");
    refreshMarkers();
    updateUI();
  });

  ui.theme.addEventListener("change", () => {
    if (renderer) {
      renderer.setTheme(ui.theme.value);
    }
  });

  ui.boardSize.addEventListener("change", () => resetGame());
  ui.humanColor.addEventListener("change", () => resetGame());
  ui.agent.addEventListener("change", () => resetGame());

  await resetGame();
}

init();
