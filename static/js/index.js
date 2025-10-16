document.addEventListener('DOMContentLoaded', () => {
  const explorerRoot = document.getElementById('trajectory-explorer');
  if (!explorerRoot) {
    return;
  }

  const elements = {
    status: document.getElementById('trajectory-status'),
    select: document.getElementById('trajectory-game'),
    viewCaptain: document.getElementById('view-captain'),
    viewSpotter: document.getElementById('view-spotter'),
    metrics: document.getElementById('trajectory-metrics'),
    board: document.getElementById('trajectory-board'),
    boardCaption: document.getElementById('board-caption'),
    slider: document.getElementById('trajectory-slider'),
    frameLabel: document.getElementById('frame-label'),
    eventLabel: document.getElementById('event-label'),
    eventDetails: document.getElementById('event-details'),
    progressDetails: document.getElementById('progress-details'),
    shipTracker: document.getElementById('ship-tracker'),
    timelineList: document.getElementById('trajectory-event-list'),
  };

  const colorPalette = new Map([
    [-1, '#eaeae4'],
    [0, '#9b9c97'],
    [1, '#ac2028'],
    [2, '#04af70'],
    [3, '#6d467b'],
    [4, '#ffa500'],
  ]);
  const fallbackColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#2b908f', '#f45b5b', '#91e8e1', '#6f38a0'];
  const shipNames = { 1: 'Red ship', 2: 'Green ship', 3: 'Purple ship', 4: 'Orange ship' };
  const shipSymbols = { 1: 'R', 2: 'G', 3: 'P', 4: 'O' };

  let fallbackIndex = 0;
  let timelineButtons = [];

  const state = {
    data: null,
    currentGameIndex: 0,
    currentStage: 0,
    currentView: 'captain',
  };

  function setStatusMessage(message, isError = false) {
    if (!elements.status) return;
    elements.status.textContent = message;
    elements.status.classList.toggle('has-error', isError);
  }

  function updateStatusForGame(game) {
    if (!elements.status || !game) return;
    const parts = [
      `${game.captain_type} vs ${game.spotter_type}`,
      `Board ${game.board_id}`,
      `Round ${game.round_id}`,
    ];
    if (typeof game.seed === 'number') {
      parts.push(`Seed ${game.seed}`);
    }
    elements.status.textContent = parts.join(' • ');
    elements.status.classList.remove('has-error');
  }

  function getColor(value) {
    const numeric = typeof value === 'number' ? value : Number(value);
    if (colorPalette.has(numeric)) {
      return colorPalette.get(numeric);
    }
    if (Number.isFinite(numeric) && numeric > 0) {
      const color = fallbackColors[fallbackIndex % fallbackColors.length];
      fallbackIndex += 1;
      colorPalette.set(numeric, color);
      return color;
    }
    return '#d0d4dc';
  }

  function formatDecimal(value, digits = 2) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      return '–';
    }
    return value.toFixed(digits);
  }

  function truncate(text, maxLength = 70) {
    if (!text) return '';
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
  }

  function coordsToTile(coords) {
    if (!Array.isArray(coords) || coords.length < 2) {
      return '';
    }
    const [row, col] = coords;
    return `${String.fromCharCode(65 + row)}${col + 1}`;
  }

  function getShipName(value) {
    return shipNames[value] || `Ship ${value}`;
  }

  function getShipSymbol(value) {
    return shipSymbols[value] || String(value);
  }

  function countCells(board, predicate) {
    if (!Array.isArray(board)) return 0;
    let total = 0;
    board.forEach((row) => {
      if (!Array.isArray(row)) return;
      row.forEach((cell) => {
        if (predicate(cell)) {
          total += 1;
        }
      });
    });
    return total;
  }

  function getCurrentGame() {
    return state.data?.games?.[state.currentGameIndex] ?? null;
  }

  function getCurrentEvent() {
    const game = getCurrentGame();
    if (!game || !Array.isArray(game.events)) return null;
    return game.events[state.currentStage] ?? null;
  }

  function populateGameOptions(games) {
    if (!elements.select) return;
    elements.select.innerHTML = '';
    games.forEach((game, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      const f1 = typeof game.f1_score === 'number' ? formatDecimal(game.f1_score) : '–';
      const outcome = game.is_won ? 'Win' : 'Loss';
      option.textContent = `#${index + 1} — ${game.captain_type} (${outcome}, F1 ${f1})`;
      elements.select.appendChild(option);
    });
  }

  function renderMetrics(game) {
    if (!elements.metrics) return;
    const metrics = [
      { label: 'Outcome', value: game.is_won ? 'Win' : 'Loss' },
      { label: 'F1 Score', value: formatDecimal(game.f1_score) },
      { label: 'Questions', value: `${game.question_count ?? '–'}` },
      { label: 'Moves', value: `${game.move_count ?? '–'}` },
    ];
    elements.metrics.innerHTML = '';
    metrics.forEach((metric) => {
      const card = document.createElement('div');
      card.className = 'metric-card';
      const label = document.createElement('span');
      label.textContent = metric.label;
      const value = document.createElement('strong');
      value.textContent = metric.value;
      card.appendChild(label);
      card.appendChild(value);
      elements.metrics.appendChild(card);
    });
  }

  function updateSlider(game) {
    if (!elements.slider) return;
    const maxStage = Math.max((game.events?.length ?? 1) - 1, 0);
    elements.slider.max = String(maxStage);
    elements.slider.setAttribute('aria-valuemax', String(maxStage));
    elements.slider.value = String(state.currentStage);
    elements.slider.setAttribute('aria-valuenow', String(state.currentStage));
    elements.slider.disabled = maxStage === 0;
  }

  function buildTimeline(game) {
    if (!elements.timelineList) return;
    timelineButtons = [];
    elements.timelineList.innerHTML = '';
    if (!Array.isArray(game.events)) return;
    let activeButton = null;
    game.events.forEach((event, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'timeline-item';
      button.dataset.stage = String(index);

      const indexBadge = document.createElement('span');
      indexBadge.className = 'timeline-index';
      indexBadge.textContent = String(index + 1);

      const summaryWrapper = document.createElement('span');
      summaryWrapper.className = 'timeline-summary';

      const typeSpan = document.createElement('span');
      typeSpan.className = 'timeline-type';
      const typeLabel = event.decision === 'move' ? 'Shot' : event.decision === 'question' ? 'Question' : (event.decision ?? 'Event');
      typeSpan.textContent = typeLabel;

      const textSpan = document.createElement('span');
      textSpan.className = 'timeline-text';
      textSpan.textContent = createTimelineText(event, game);

      summaryWrapper.appendChild(typeSpan);
      summaryWrapper.appendChild(textSpan);

      button.appendChild(indexBadge);
      button.appendChild(summaryWrapper);

      button.addEventListener('click', () => {
        setStage(index);
      });

      elements.timelineList.appendChild(button);
      timelineButtons.push(button);

      if (index === state.currentStage) {
        activeButton = button;
      }
    });

    if (activeButton) {
      requestAnimationFrame(() => {
        activeButton.scrollIntoView({ block: 'nearest' });
      });
    }
  }

  function highlightActiveTimeline() {
    timelineButtons.forEach((button, index) => {
      const isActive = index === state.currentStage;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      if (isActive) {
        requestAnimationFrame(() => {
          button.scrollIntoView({ block: 'nearest' });
        });
      }
    });
  }

  function renderBoard(game, event) {
    if (!elements.board) return;
    const trueBoard = game.true_board;
    const partialBoard = event.board;
    if (!Array.isArray(trueBoard) || !Array.isArray(partialBoard)) {
      elements.board.innerHTML = '<p>Board unavailable.</p>';
      return;
    }
    const size = trueBoard.length;
    elements.board.innerHTML = '';
    elements.board.style.setProperty('--board-columns', String(size + 1));

    const topLeft = document.createElement('div');
    topLeft.className = 'board-label-cell';
    topLeft.setAttribute('role', 'presentation');
    elements.board.appendChild(topLeft);

    for (let col = 0; col < size; col += 1) {
      const header = document.createElement('div');
      header.className = 'board-label-cell';
      header.textContent = String(col + 1);
      header.setAttribute('role', 'columnheader');
      elements.board.appendChild(header);
    }

    const currentCoords = event.decision === 'move' ? event.move?.coords : null;

    for (let row = 0; row < size; row += 1) {
      const rowLabel = document.createElement('div');
      rowLabel.className = 'board-label-cell';
      rowLabel.textContent = String.fromCharCode(65 + row);
      rowLabel.setAttribute('role', 'rowheader');
      elements.board.appendChild(rowLabel);

      for (let col = 0; col < size; col += 1) {
        const cell = document.createElement('div');
        cell.className = 'board-cell';
        const partialValue = partialBoard[row]?.[col];
        const trueValue = trueBoard[row]?.[col];
        const isCurrent = Array.isArray(currentCoords) && currentCoords[0] === row && currentCoords[1] === col;

        const displayValue = state.currentView === 'captain' ? partialValue : trueValue;
        cell.style.setProperty('--cell-bg', getColor(displayValue));
        if (state.currentView === 'spotter' && partialValue === -1) {
          cell.setAttribute('data-hidden', 'true');
        } else {
          cell.removeAttribute('data-hidden');
        }

        if (state.currentView === 'captain') {
          if (partialValue === 0) {
            cell.textContent = '•';
          } else if (partialValue > 0) {
            cell.textContent = getShipSymbol(partialValue);
          }
        } else if (trueValue > 0) {
          cell.textContent = getShipSymbol(trueValue);
        }

        if (isCurrent) {
          cell.classList.add('is-current');
        }

        cell.setAttribute('aria-label', `${String.fromCharCode(65 + row)}${col + 1}`);
        elements.board.appendChild(cell);
      }
    }
  }

  function computeShipSummary(trueBoard, partialBoard) {
    const summary = new Map();
    trueBoard.forEach((row, rowIndex) => {
      row.forEach((value, colIndex) => {
        if (value > 0) {
          const entry = summary.get(value) ?? { id: value, total: 0, revealed: 0 };
          entry.total += 1;
          if (partialBoard[rowIndex]?.[colIndex] === value) {
            entry.revealed += 1;
          }
          summary.set(value, entry);
        }
      });
    });
    return Array.from(summary.values()).sort((a, b) => a.id - b.id);
  }

  function renderShipTracker(game, partialBoard) {
    if (!elements.shipTracker) return;
    elements.shipTracker.innerHTML = '';
    if (!Array.isArray(game.true_board)) {
      elements.shipTracker.textContent = 'Ship data unavailable.';
      return;
    }
    const ships = computeShipSummary(game.true_board, partialBoard);
    if (!ships.length) {
      elements.shipTracker.textContent = 'No ships detected on this board.';
      return;
    }
    ships.forEach((ship) => {
      const row = document.createElement('div');
      row.className = 'ship-row';

      const dot = document.createElement('span');
      dot.className = 'ship-color-dot';
      dot.style.background = getColor(ship.id);

      const label = document.createElement('span');
      label.className = 'ship-label';
      label.textContent = `${getShipName(ship.id)} (${ship.total})`;

      const progress = document.createElement('div');
      progress.className = 'ship-progress';

      const progressBar = document.createElement('div');
      progressBar.className = 'ship-progress-bar';
      progressBar.style.background = getColor(ship.id);
      const percentage = ship.total === 0 ? 0 : Math.round((ship.revealed / ship.total) * 100);
      progressBar.style.width = `${percentage}%`;

      const status = document.createElement('span');
      status.className = 'ship-status';
      status.textContent = `${ship.revealed}/${ship.total}${ship.revealed === ship.total ? ' ✓' : ''}`;

      progress.appendChild(progressBar);
      row.appendChild(dot);
      row.appendChild(label);
      row.appendChild(progress);
      row.appendChild(status);

      elements.shipTracker.appendChild(row);
    });
  }

  function renderEventDetails(game, event) {
    if (!elements.eventDetails) return;
    elements.eventDetails.innerHTML = '';

    const typeLabel = document.createElement('div');
    typeLabel.className = 'event-type';
    const typeText = event.decision === 'move' ? 'Move' : event.decision === 'question' ? 'Question' : 'Event';
    typeLabel.textContent = typeText;
    elements.eventDetails.appendChild(typeLabel);

    if (event.decision === 'move') {
      const move = document.createElement('div');
      move.className = 'event-shot';
      const coords = event.move?.coords;
      const tile = event.move?.tile || coordsToTile(coords);
      const shipValue = Array.isArray(coords) ? game.true_board?.[coords[0]]?.[coords[1]] : null;
      const hit = typeof shipValue === 'number' && shipValue > 0 && event.board?.[coords[0]]?.[coords[1]] === shipValue;
      const result = hit ? `Hit ${getShipName(shipValue)}` : 'Missed water';
      move.textContent = `Shot at ${tile} — ${result}`;
      elements.eventDetails.appendChild(move);
    } else if (event.decision === 'question') {
      if (event.question?.text) {
        const question = document.createElement('div');
        question.className = 'event-question';
        question.textContent = event.question.text;
        elements.eventDetails.appendChild(question);
      }
      if (event.answer) {
        const answer = document.createElement('div');
        answer.className = 'event-answer';
        const answerText = typeof event.answer.text === 'string'
          ? event.answer.text
          : typeof event.answer.value === 'boolean'
            ? event.answer.value ? 'yes' : 'no'
            : String(event.answer.value ?? '');
        answer.textContent = `Answer: ${answerText}`;
        elements.eventDetails.appendChild(answer);
      }
    } else {
      const note = document.createElement('div');
      note.className = 'event-shot';
      note.textContent = 'No additional details for this event.';
      elements.eventDetails.appendChild(note);
    }
  }

  function renderProgress(game, partialBoard) {
    if (!elements.progressDetails) return;
    const eventsSoFar = game.events.slice(0, state.currentStage + 1);
    const movesSoFar = eventsSoFar.filter((event) => event.decision === 'move').length;
    const questionsSoFar = eventsSoFar.filter((event) => event.decision === 'question').length;
    const hitsSoFar = countCells(partialBoard, (value) => typeof value === 'number' && value > 0);
    const missesSoFar = countCells(partialBoard, (value) => value === 0);

    const progressEntries = [
      { label: 'Moves', value: `${movesSoFar}/${game.move_count ?? '–'}` },
      { label: 'Questions', value: `${questionsSoFar}/${game.question_count ?? '–'}` },
      { label: 'Hits', value: `${hitsSoFar}/${game.hits ?? '–'}` },
      { label: 'Misses', value: `${missesSoFar}/${game.misses ?? '–'}` },
    ];

    elements.progressDetails.innerHTML = '';
    const grid = document.createElement('div');
    grid.className = 'progress-grid';
    progressEntries.forEach((entry) => {
      const chip = document.createElement('div');
      chip.className = 'progress-chip';
      chip.textContent = `${entry.label}: ${entry.value}`;
      grid.appendChild(chip);
    });
    elements.progressDetails.appendChild(grid);
  }

  function createTimelineText(event, game) {
    if (event.decision === 'move') {
      const coords = event.move?.coords;
      const tile = event.move?.tile || coordsToTile(coords);
      const shipValue = Array.isArray(coords) ? game.true_board?.[coords[0]]?.[coords[1]] : null;
      const hit = typeof shipValue === 'number' && shipValue > 0 && event.board?.[coords[0]]?.[coords[1]] === shipValue;
      const result = hit ? `Hit ${getShipName(shipValue)}` : 'Miss';
      return `Shot at ${tile} → ${result}`;
    }
    if (event.decision === 'question') {
      const questionText = event.question?.text ? truncate(event.question.text, 60) : 'Question';
      const answerText = event.answer
        ? truncate(
            typeof event.answer.text === 'string'
              ? event.answer.text
              : String(event.answer.value ?? ''),
            20,
          )
        : '–';
      return `${questionText} → ${answerText}`;
    }
    return event.decision ? `${event.decision}` : 'Event';
  }

  function createEventSummary(event, game) {
    if (event.decision === 'move') {
      const coords = event.move?.coords;
      const tile = event.move?.tile || coordsToTile(coords);
      const shipValue = Array.isArray(coords) ? game.true_board?.[coords[0]]?.[coords[1]] : null;
      const hit = typeof shipValue === 'number' && shipValue > 0 && event.board?.[coords[0]]?.[coords[1]] === shipValue;
      if (hit) {
        return `Shot ${tile} and hit ${getShipName(shipValue)}.`;
      }
      return `Shot ${tile} and missed.`;
    }
    if (event.decision === 'question') {
      const questionText = event.question?.text || 'Question';
      const answerText = event.answer
        ? typeof event.answer.text === 'string'
          ? event.answer.text
          : typeof event.answer.value === 'boolean'
            ? event.answer.value ? 'yes' : 'no'
            : String(event.answer.value ?? '')
        : 'no answer';
      return `Asked “${questionText}” → ${answerText}.`;
    }
    return `Event: ${event.decision ?? 'N/A'}.`;
  }

  function updateFrameAndEventLabels(game, event) {
    if (elements.frameLabel) {
      elements.frameLabel.textContent = `Turn ${state.currentStage + 1} of ${game.events.length}`;
    }
    if (elements.eventLabel) {
      elements.eventLabel.textContent = createEventSummary(event, game);
    }
    if (elements.boardCaption) {
      if (state.currentView === 'captain') {
        elements.boardCaption.textContent = `Captain view after turn ${state.currentStage + 1}`;
      } else {
        elements.boardCaption.textContent = 'Spotter view (hatched = hidden from the captain)';
      }
    }
  }

  function renderStageDependent() {
    const game = getCurrentGame();
    const event = getCurrentEvent();
    if (!game || !event) return;
    renderBoard(game, event);
    renderShipTracker(game, event.board);
    renderEventDetails(game, event);
    renderProgress(game, event.board);
    updateFrameAndEventLabels(game, event);
    highlightActiveTimeline();
    if (elements.slider) {
      elements.slider.setAttribute('aria-valuenow', String(state.currentStage));
    }
  }

  function setGame(index) {
    const games = state.data?.games;
    if (!Array.isArray(games) || games.length === 0) return;
    const clampedIndex = Math.max(0, Math.min(index, games.length - 1));
    state.currentGameIndex = clampedIndex;
    state.currentStage = 0;
    if (elements.select) {
      elements.select.value = String(clampedIndex);
    }
    const game = getCurrentGame();
    if (!game) return;
    updateStatusForGame(game);
    renderMetrics(game);
    buildTimeline(game);
    updateSlider(game);
    renderStageDependent();
  }

  function setStage(stage) {
    const game = getCurrentGame();
    if (!game || !Array.isArray(game.events) || game.events.length === 0) return;
    const clampedStage = Math.max(0, Math.min(stage, game.events.length - 1));
    state.currentStage = clampedStage;
    if (elements.slider) {
      elements.slider.value = String(clampedStage);
    }
    renderStageDependent();
  }

  function updateViewButtons() {
    if (elements.viewCaptain) {
      const isActive = state.currentView === 'captain';
      elements.viewCaptain.classList.toggle('is-active', isActive);
      elements.viewCaptain.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    }
    if (elements.viewSpotter) {
      const isActive = state.currentView === 'spotter';
      elements.viewSpotter.classList.toggle('is-active', isActive);
      elements.viewSpotter.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    }
  }

  function setView(view) {
    if (state.currentView === view) return;
    state.currentView = view;
    updateViewButtons();
    renderStageDependent();
  }

  function attachEventListeners() {
    if (elements.select) {
      elements.select.addEventListener('change', (event) => {
        const index = Number(event.target.value);
        setGame(Number.isNaN(index) ? 0 : index);
      });
    }

    if (elements.slider) {
      elements.slider.addEventListener('input', (event) => {
        const stage = Number(event.target.value);
        setStage(Number.isNaN(stage) ? 0 : stage);
      });
    }

    if (elements.viewCaptain) {
      elements.viewCaptain.addEventListener('click', () => setView('captain'));
    }

    if (elements.viewSpotter) {
      elements.viewSpotter.addEventListener('click', () => setView('spotter'));
    }
  }

  async function loadData() {
    try {
      setStatusMessage('Loading curated games…');
      const response = await fetch('./static/data/trajectory_samples.json', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (!Array.isArray(data.games) || data.games.length === 0) {
        setStatusMessage('No trajectory samples found.', true);
        return;
      }
      state.data = data;
      populateGameOptions(data.games);
      attachEventListeners();
      updateViewButtons();
      setGame(0);
    } catch (error) {
      console.error('Failed to load trajectory samples', error);
      setStatusMessage('Unable to load trajectory samples. Please refresh the page.', true);
    }
  }

  loadData();
});
