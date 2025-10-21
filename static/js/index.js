document.addEventListener('DOMContentLoaded', () => {
  const explorerRoot = document.getElementById('trajectory-explorer');
  if (!explorerRoot) {
    return;
  }

  const elements = {
    status: document.getElementById('trajectory-status'),
    filterLLM: document.getElementById('filter-llm'),
    filterType: document.getElementById('filter-type'),
    gameList: document.getElementById('trajectory-game-list'),
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
    playButton: document.getElementById('timeline-play'),
    timelineReset: document.getElementById('timeline-reset'),
    timelineSkip: document.getElementById('timeline-skip'),
    prevGame: document.getElementById('prev-game'),
    nextGame: document.getElementById('next-game'),
    speedSelect: document.getElementById('playback-speed'),
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
  // Invert symbol mapping for convenience: 'R' -> 1, etc.
  const shipIdBySymbol = Object.fromEntries(Object.entries(shipSymbols).map(([id, sym]) => [sym, Number(id)]));

  const DEFAULT_PLAYBACK_INTERVAL_MS = 1000;

  const LLM_ORDER = ['GPT-5', 'GPT-4o', 'Llama-4-Scout', 'Baseline'];

  const LLM_LABELS = {
    'GPT-5': 'GPT-5',
    'GPT-4o': 'GPT-4o',
    'Llama-4-Scout': 'Llama-4-Scout',
    'Baseline': 'Baseline (No LLM)',
  };

  const CAPTAIN_TYPE_ORDER = ['LM', '+Bayes-Q', '+Bayes-M', '+Bayes-QM', '+Bayes-QMD', 'Random', 'Greedy'];

  const CAPTAIN_TYPE_LABELS = {
    'LM': 'LM-only',
    '+Bayes-Q': 'LM+Bayes-Q',
    '+Bayes-M': 'LM+Bayes-M',
    '+Bayes-QM': 'LM+Bayes-QM',
    '+Bayes-QMD': 'LM+Bayes-QMD',
    'Random': 'Random',
    'Greedy': 'Greedy',
  };

  let fallbackIndex = 0;
  let gameButtons = new Map();
  let timelineButtons = [];
  let timelineScrollRequested = false;

  const state = {
    data: null,
    currentGameIndex: 0,
    currentStage: 0,
    currentView: 'captain',
    filters: {
      llm: '',
      type: '',
    },
    filteredIndices: [],
    hasActiveGame: false,
    playbackSpeed: 1,
  };

  const playback = {
    intervalMs: DEFAULT_PLAYBACK_INTERVAL_MS,
    timerId: null,
    isPlaying: false,
    sequenceId: 0,
  };

  const sliderAnimation = {
    frameId: null,
    startValue: 0,
    targetValue: 0,
    startTime: 0,
    duration: 0,
    onComplete: null,
  };

  const SLIDER_ANIMATION_DURATION_RATIO = 1.0;

  const sliderStepControl = {
    defaultStep: elements.slider ? elements.slider.getAttribute('step') || '1' : '1',
    isStepless: false,
  };

  const TYPEWRITER_CHAR_DELAY_MS = 26;
  const TYPEWRITER_PUNCTUATION_DELAY_MS = 140;
  const TYPEWRITER_SPACE_DELAY_MS = 35;
  const THINKING_BASE_DELAY_MS = 520;
  const RESULT_REVEAL_DELAY_MS = 220;
  const QUESTION_LINGER_BUFFER_MS = 900;
  const MOVE_LINGER_BUFFER_MS = 320;
  const OTHER_LINGER_BUFFER_MS = 500;
  let activeEventAnimationId = 0;
  let lastEventAnimationPromise = Promise.resolve();
  let lastRenderedStageIndex = 0;

  const ICON_CLASSES = {
    captain: 'fas fa-anchor',
    spotter: 'fas fa-binoculars',
    miss: 'fas fa-times',
    hit: 'fas fa-bullseye',
  };

  function prefersReducedMotion() {
    return Boolean(window.matchMedia) && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function setTypingCaretVisibility(caret, isVisible) {
    if (!caret) return;
    caret.classList.toggle('is-visible', Boolean(isVisible));
  }

  function getAvatarIconClass(role) {
    if (role === 'captain') return ICON_CLASSES.captain;
    if (role === 'spotter') return ICON_CLASSES.spotter;
    return '';
  }

  function delayWithCancel(durationMs, animationId, scale = 1) {
    const duration = prefersReducedMotion() ? 0 : Math.max(0, durationMs * scale);
    if (duration === 0) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const timeoutId = window.setTimeout(() => {
        resolve();
      }, duration);
      if (animationId !== activeEventAnimationId) {
        window.clearTimeout(timeoutId);
        resolve();
      }
    });
  }

  function formatAnswerText(answer) {
    if (!answer) return 'Awaiting response…';
    if (typeof answer.text === 'string' && answer.text.trim().length) {
      return answer.text.trim();
    }
    if (typeof answer.value === 'boolean') {
      return answer.value ? 'Yes.' : 'No.';
    }
    if (answer.value !== undefined && answer.value !== null) {
      return String(answer.value);
    }
    return 'Awaiting response…';
  }

  function createThinkingElement() {
    const thinker = document.createElement('span');
    thinker.className = 'event-thinking';
    thinker.setAttribute('aria-hidden', 'true');
    for (let i = 0; i < 3; i += 1) {
      const dot = document.createElement('span');
      dot.className = 'event-thinking-dot';
      dot.setAttribute('aria-hidden', 'true');
      thinker.appendChild(dot);
    }
    return thinker;
  }

  function buildResultElement({ isHit, text }) {
    const result = document.createElement('div');
    result.className = 'event-result';
    result.classList.add(isHit ? 'is-hit' : 'is-miss');
    const icon = document.createElement('i');
    icon.className = `result-icon ${isHit ? ICON_CLASSES.hit : ICON_CLASSES.miss}`;
    icon.setAttribute('aria-hidden', 'true');
    const copy = document.createElement('span');
    copy.textContent = text;
    result.appendChild(icon);
    result.appendChild(copy);
    return result;
  }

  function computeLingerDuration(event) {
    const base = playback.intervalMs;
    let buffer = OTHER_LINGER_BUFFER_MS;
    if (event?.decision === 'question') {
      buffer = QUESTION_LINGER_BUFFER_MS;
    } else if (event?.decision === 'move') {
      buffer = MOVE_LINGER_BUFFER_MS;
    }
    const speed = state.playbackSpeed || 1;
    const adjustedBuffer = prefersReducedMotion() ? 0 : buffer / speed;
    return Math.max(0, base + adjustedBuffer);
  }

  async function displayTypewrittenMessage({
    logEl,
    role,
    label,
    text,
    animationId,
  }) {
    if (!logEl) return null;
    const message = createChatMessage({ role, label });
    logEl.appendChild(message.wrapper);
    logEl.scrollTop = logEl.scrollHeight;
    await typeText({
      textEl: message.textEl,
      caret: message.caret,
      text,
      animationId,
      logEl,
    });
    return message;
  }

  function createChatMessage({ role, label, iconClass }) {
    const messageWrapper = document.createElement('div');
    messageWrapper.className = 'event-message';

    if (role === 'captain') {
      messageWrapper.classList.add('is-captain');
    } else if (role === 'spotter') {
      messageWrapper.classList.add('is-spotter');
    } else {
      messageWrapper.classList.add('is-system');
    }

    let avatarEl = null;
    if (role !== 'system') {
      avatarEl = document.createElement('div');
      avatarEl.className = 'event-avatar';
      avatarEl.setAttribute('aria-hidden', 'true');
      const resolvedIconClass = iconClass || getAvatarIconClass(role);
      if (resolvedIconClass) {
        const iconEl = document.createElement('i');
        iconEl.className = resolvedIconClass;
        iconEl.setAttribute('aria-hidden', 'true');
        avatarEl.appendChild(iconEl);
      } else {
        avatarEl.textContent = (label || role || '?').trim().charAt(0).toUpperCase() || '•';
      }
      messageWrapper.appendChild(avatarEl);
    }

    const bubble = document.createElement('div');
    bubble.className = 'event-bubble';

    const speaker = document.createElement('span');
    speaker.className = 'event-speaker';
    speaker.textContent = label || 'Update';
    bubble.appendChild(speaker);

    const textEl = document.createElement('span');
    textEl.className = 'event-message-text';
    bubble.appendChild(textEl);

    const caret = document.createElement('span');
    caret.className = 'event-typing-caret';
    bubble.appendChild(caret);

    messageWrapper.appendChild(bubble);

    return {
      wrapper: messageWrapper,
      avatar: avatarEl,
      textEl,
      caret,
      bubble,
    };
  }

  function typeText({ textEl, caret, text, animationId, logEl, timeScale = 1 }) {
    return new Promise((resolve) => {
      const messageText = typeof text === 'string' ? text : '';
      if (!textEl) {
        resolve();
        return;
      }

      if (prefersReducedMotion() || animationId !== activeEventAnimationId || messageText.length === 0) {
        textEl.textContent = messageText;
        setTypingCaretVisibility(caret, false);
        if (logEl) {
          logEl.scrollTop = logEl.scrollHeight;
        }
        resolve();
        return;
      }

      textEl.textContent = '';
      setTypingCaretVisibility(caret, true);
      let index = 0;

      const step = () => {
        if (animationId !== activeEventAnimationId) {
          textEl.textContent = messageText;
          setTypingCaretVisibility(caret, false);
          if (logEl) {
            logEl.scrollTop = logEl.scrollHeight;
          }
          resolve();
          return;
        }

        textEl.textContent += messageText.charAt(index);
        index += 1;

        if (logEl) {
          logEl.scrollTop = logEl.scrollHeight;
        }

        if (index >= messageText.length) {
          setTypingCaretVisibility(caret, false);
          resolve();
          return;
        }

        const previousChar = messageText.charAt(index - 1);
        const scale = Number.isFinite(timeScale) && timeScale > 0 ? timeScale : 1;
        let delay = TYPEWRITER_CHAR_DELAY_MS;
        if (/[,.;!?]/.test(previousChar)) {
          delay = TYPEWRITER_PUNCTUATION_DELAY_MS;
        } else if (previousChar === ' ') {
          delay = TYPEWRITER_SPACE_DELAY_MS;
        }

        window.setTimeout(step, delay * scale);
      };

      step();
    });
  }

  function getSliderMaxValue() {
    if (!elements.slider) return 0;
    const maxAttr = Number(elements.slider.max);
    if (Number.isFinite(maxAttr)) {
      return maxAttr;
    }
    const ariaMax = Number(elements.slider.getAttribute('aria-valuemax'));
    return Number.isFinite(ariaMax) ? ariaMax : 0;
  }

  function updateSliderProgress(value) {
    if (!elements.slider) return;
    const max = getSliderMaxValue();
    const numericValue = Number(value);
    let percent = 0;
    if (Number.isFinite(numericValue) && max > 0) {
      percent = Math.min(Math.max(numericValue / max, 0), 1) * 100;
    }
    elements.slider.style.setProperty('--slider-progress', `${percent}%`);
  }

  function setSliderValue(value) {
    if (!elements.slider) return;
    const numericValue = Number(value);
    const safeValue = Number.isFinite(numericValue) ? numericValue : 0;
    elements.slider.value = String(safeValue);
    updateSliderProgress(safeValue);
  }

  function enableSliderStepless() {
    const slider = elements.slider;
    if (!slider || sliderStepControl.isStepless) return;
    slider.setAttribute('step', 'any');
    sliderStepControl.isStepless = true;
  }

  function disableSliderStepless() {
    const slider = elements.slider;
    if (!slider) return;
    if (sliderStepControl.defaultStep) {
      slider.setAttribute('step', sliderStepControl.defaultStep);
    } else {
      slider.removeAttribute('step');
    }
    sliderStepControl.isStepless = false;
  }

  function setStatusMessage(message, isError = false) {
    if (!elements.status) return;
    elements.status.textContent = message;
    elements.status.classList.toggle('has-error', isError);
  }

  function updateStatusForGame(game) {
    if (!elements.status || !game) return;
    const parts = [
      `${game.captain_llm}`,
      `${game.captain_type}`,
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

  function updatePlayButton() {
    if (!elements.playButton) return;
    const button = elements.playButton;
    const icon = button.querySelector('.playback-icon');
    const label = button.querySelector('.playback-text');
    if (playback.isPlaying) {
      button.classList.remove('is-paused');
      button.setAttribute('aria-label', 'Pause autoplay');
      button.setAttribute('aria-pressed', 'true');
      if (icon) {
        icon.classList.remove('fa-play');
        icon.classList.add('fa-pause');
      }
      if (label) label.textContent = 'Pause';
    } else {
      button.classList.add('is-paused');
      button.setAttribute('aria-label', 'Start autoplay');
      button.setAttribute('aria-pressed', 'false');
      if (icon) {
        icon.classList.remove('fa-pause');
        icon.classList.add('fa-play');
      }
      if (label) label.textContent = 'Play';
    }
  }

  function clearPlaybackTimer() {
    playback.sequenceId += 1;
    if (playback.timerId !== null) {
      clearTimeout(playback.timerId);
      playback.timerId = null;
    }
  }

  function cancelSliderAnimation() {
    if (sliderAnimation.frameId !== null) {
      window.cancelAnimationFrame(sliderAnimation.frameId);
      sliderAnimation.frameId = null;
    }
    sliderAnimation.onComplete = null;
  }

  function easeOutCubic(t) {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    const delta = 1 - t;
    return 1 - delta * delta * delta;
  }

  function animateSliderValue(startValue, targetValue, duration, onComplete) {
    const slider = elements.slider;
    if (!slider) return;
    cancelSliderAnimation();
    let effectiveDuration = duration;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      effectiveDuration = 0;
    }
    sliderAnimation.startValue = startValue;
    sliderAnimation.targetValue = targetValue;
    sliderAnimation.duration = effectiveDuration;
    sliderAnimation.startTime = performance.now();
    sliderAnimation.onComplete = typeof onComplete === 'function' ? onComplete : null;

    if (effectiveDuration <= 0 || startValue === targetValue) {
      setSliderValue(targetValue);
      sliderAnimation.frameId = null;
      if (sliderAnimation.onComplete) {
        sliderAnimation.onComplete();
        sliderAnimation.onComplete = null;
      }
      return;
    }

    setSliderValue(startValue);

    const step = (now) => {
      const elapsed = now - sliderAnimation.startTime;
  const progress = Math.min(elapsed / sliderAnimation.duration, 1);
  const eased = easeOutCubic(progress);
      const value = sliderAnimation.startValue + (sliderAnimation.targetValue - sliderAnimation.startValue) * eased;
      setSliderValue(value);
      if (progress < 1) {
        sliderAnimation.frameId = window.requestAnimationFrame(step);
      } else {
        sliderAnimation.frameId = null;
        setSliderValue(targetValue);
        if (sliderAnimation.onComplete) {
          sliderAnimation.onComplete();
          sliderAnimation.onComplete = null;
        }
      }
    };

    sliderAnimation.frameId = window.requestAnimationFrame(step);
  }

  function schedulePlaybackAfterEvent(animationPromise, game, stageIndex) {
    clearPlaybackTimer();
    if (!playback.isPlaying) {
      return;
    }
    if (!game || !Array.isArray(game.events) || game.events.length === 0) {
      return;
    }
    if (stageIndex >= game.events.length - 1) {
      playback.isPlaying = false;
      updatePlayButton();
      if (sliderAnimation.frameId === null) {
        disableSliderStepless();
      }
      return;
    }

    const sequenceId = ++playback.sequenceId;
    Promise.resolve(animationPromise)
      .catch((error) => {
        console.error('Event animation failed', error);
      })
      .then(() => {
        if (!playback.isPlaying || playback.sequenceId !== sequenceId) {
          return;
        }
        if (state.currentStage !== stageIndex) {
          return;
        }

        const lingerDuration = computeLingerDuration(game.events[stageIndex]);
        const sliderDuration = playback.intervalMs * SLIDER_ANIMATION_DURATION_RATIO;
        playback.timerId = window.setTimeout(() => {
          if (!playback.isPlaying || playback.sequenceId !== sequenceId) {
            return;
          }
          if (state.currentStage !== stageIndex) {
            return;
          }
          const nextStage = stageIndex + 1;
          setStage(nextStage, {
            scrollTimeline: false,
            animateSlider: true,
            animationDuration: sliderDuration,
          });
        }, lingerDuration);
      });
  }

  function computeIntervalForSpeed(speed) {
    const multiplier = Number(speed);
    if (!Number.isFinite(multiplier) || multiplier <= 0) {
      return DEFAULT_PLAYBACK_INTERVAL_MS;
    }
    return DEFAULT_PLAYBACK_INTERVAL_MS / multiplier;
  }

  function updateSpeedControl() {
    if (!elements.speedSelect) return;
    const value = String(state.playbackSpeed);
    if (elements.speedSelect.value !== value) {
      elements.speedSelect.value = value;
    }
  }

  function setPlaybackSpeed(speed) {
    const multiplier = Number(speed);
    if (!Number.isFinite(multiplier) || multiplier <= 0) {
      return;
    }
    if (Math.abs(multiplier - state.playbackSpeed) < 1e-6) {
      updateSpeedControl();
      return;
    }
    state.playbackSpeed = multiplier;
    playback.intervalMs = computeIntervalForSpeed(multiplier);
    clearPlaybackTimer();
    updateSpeedControl();
    if (playback.isPlaying) {
      schedulePlaybackAfterEvent(lastEventAnimationPromise, getCurrentGame(), lastRenderedStageIndex);
    }
  }

  function setButtonAvailability(button, enabled) {
    if (!button) return;
    button.disabled = !enabled;
    button.setAttribute('aria-disabled', enabled ? 'false' : 'true');
  }

  function updateGameNavButtons() {
    if (!elements.prevGame || !elements.nextGame) return;
    const position = state.filteredIndices.indexOf(state.currentGameIndex);
    const hasGames = position !== -1 && state.filteredIndices.length > 0;
    const canGoPrev = hasGames && position > 0;
    const canGoNext = hasGames && position < state.filteredIndices.length - 1;
    setButtonAvailability(elements.prevGame, canGoPrev);
    setButtonAvailability(elements.nextGame, canGoNext);
  }

  function changeGame(offset) {
    if (!Number.isInteger(offset) || state.filteredIndices.length === 0) return;
    let position = state.filteredIndices.indexOf(state.currentGameIndex);
    if (position === -1) {
      position = 0;
    }
    const nextPosition = Math.min(
      Math.max(position + offset, 0),
      state.filteredIndices.length - 1,
    );
    if (nextPosition === position) return;
    const nextGameIndex = state.filteredIndices[nextPosition];
    setGame(nextGameIndex);
    const button = gameButtons.get(nextGameIndex);
    if (button) {
      button.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      button.focus();
    }
  }

  function setPlaying(isPlaying) {
    const next = Boolean(isPlaying);
    cancelSliderAnimation();
    if (next) {
      enableSliderStepless();
      const game = getCurrentGame();
      if (game && Array.isArray(game.events) && game.events.length > 0 && state.currentStage >= game.events.length - 1) {
        state.currentStage = 0;
        setSliderValue(0);
        timelineScrollRequested = false;
        renderStageDependent();
      }
    }
    if (next === playback.isPlaying) {
      if (next) {
        schedulePlaybackAfterEvent(lastEventAnimationPromise, getCurrentGame(), lastRenderedStageIndex);
      } else {
        clearPlaybackTimer();
      }
      return;
    }
    playback.isPlaying = next;
    updatePlayButton();
    if (playback.isPlaying) {
      schedulePlaybackAfterEvent(lastEventAnimationPromise, getCurrentGame(), lastRenderedStageIndex);
    } else {
      clearPlaybackTimer();
      disableSliderStepless();
    }
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

  // Compute ship tracker rows using the same logic as Board.ship_tracker in Python.
  // Returns an array of { length, sunkSymbol|null, id }
  function getShipTrackerRows(trueBoard, partialBoard) {
    if (!Array.isArray(trueBoard) || !Array.isArray(partialBoard)) return [];
    // Find max ship id present on the true board
    let maxId = 0;
    trueBoard.forEach((row) => {
      if (!Array.isArray(row)) return;
      row.forEach((v) => {
        const n = Number(v);
        if (Number.isFinite(n) && n > 0) maxId = Math.max(maxId, n);
      });
    });

    const rows = [];
    for (let shipType = 1; shipType <= maxId; shipType += 1) {
      // Count tiles of this ship id in true and partial boards
      let targetCount = 0;
      let stateCount = 0;
      for (let i = 0; i < trueBoard.length; i += 1) {
        const tRow = trueBoard[i] || [];
        const pRow = partialBoard[i] || [];
        for (let j = 0; j < tRow.length; j += 1) {
          if (tRow[j] === shipType) targetCount += 1;
          if (pRow[j] === shipType) stateCount += 1;
        }
      }
      if (targetCount > 0) {
        const symbol = shipSymbols[shipType] || String(shipType);
        const sunk = stateCount === targetCount; // reveal color only once fully sunk
        rows.push({ length: targetCount, sunkSymbol: sunk ? symbol : null, id: shipType });
      }
    }

    // Sort by ship length descending to match Python figure behavior
    rows.sort((a, b) => b.length - a.length);
    return rows;
  }

  function getCurrentGame() {
    return state.data?.games?.[state.currentGameIndex] ?? null;
  }

  function getCurrentEvent() {
    const game = getCurrentGame();
    if (!game || !Array.isArray(game.events)) return null;
    return game.events[state.currentStage] ?? null;
  }

  function getUniqueValues(games, key) {
    const values = new Set();
    games.forEach((game) => {
      const value = game?.[key];
      if (value !== undefined && value !== null && value !== '') {
        values.add(String(value));
      }
    });
    return Array.from(values);
  }

  function getOrderRank(value, order) {
    if (!Array.isArray(order)) return Number.POSITIVE_INFINITY;
    const index = order.indexOf(value);
    return index === -1 ? order.length : index;
  }

  function compareByOrder(a, b, order) {
    const rankA = getOrderRank(a, order);
    const rankB = getOrderRank(b, order);
    return rankA - rankB;
  }

  function sortByPreferredOrder(values, preferredOrder) {
    if (!Array.isArray(values)) return [];
    if (!Array.isArray(preferredOrder) || preferredOrder.length === 0) {
      return [...values].sort();
    }
    const valueSet = new Set(values);
    const ordered = preferredOrder.filter((value) => valueSet.has(value));
    const remaining = values.filter((value) => !preferredOrder.includes(value)).sort();
    return [...ordered, ...remaining];
  }

  function renderFilterOptions(selectEl, values, defaultLabel, labels = {}) {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = defaultLabel;
    selectEl.appendChild(defaultOption);
    values.forEach((value) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = labels[value] ?? value;
      selectEl.appendChild(option);
    });
  }

  function populateFilterOptions(games) {
    const llmValues = sortByPreferredOrder(getUniqueValues(games, 'captain_llm'), LLM_ORDER);
    const typeValues = sortByPreferredOrder(getUniqueValues(games, 'captain_type'), CAPTAIN_TYPE_ORDER);
    renderFilterOptions(elements.filterLLM, llmValues, 'All LLMs', LLM_LABELS);
    renderFilterOptions(elements.filterType, typeValues, 'All strategies', CAPTAIN_TYPE_LABELS);

    if (elements.filterLLM) {
      const desired = state.filters.llm;
      const value = desired && llmValues.includes(desired) ? desired : '';
      elements.filterLLM.value = value;
      state.filters.llm = elements.filterLLM.value || '';
    }

    if (elements.filterType) {
      const desired = state.filters.type;
      const value = desired && typeValues.includes(desired) ? desired : '';
      elements.filterType.value = value;
      state.filters.type = elements.filterType.value || '';
    }
  }

  function renderGameList() {
    if (!elements.gameList) return;
    elements.gameList.innerHTML = '';
    gameButtons = new Map();

    if (!state.filteredIndices.length) {
      const message = document.createElement('div');
      message.className = 'empty-state';
      message.textContent = 'No games match these filters.';
      elements.gameList.appendChild(message);
      elements.gameList.removeAttribute('aria-activedescendant');
      updateGameNavButtons();
      return;
    }

    state.filteredIndices.forEach((gameIndex) => {
      const game = state.data?.games?.[gameIndex];
      if (!game) return;

      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'game-item';
      button.id = `game-option-${gameIndex}`;
      button.dataset.gameIndex = String(gameIndex);
      button.setAttribute('role', 'option');

      const title = document.createElement('span');
      title.className = 'game-item-title';
      const llmLabel = LLM_LABELS[game.captain_llm] ?? game.captain_llm ?? 'Unknown LLM';
      title.textContent = String(llmLabel);

      const meta = document.createElement('span');
      meta.className = 'game-item-meta';
      const metaParts = [];
      if (game.captain_type) {
        const captainTypeLabel = CAPTAIN_TYPE_LABELS[game.captain_type] ?? game.captain_type;
        metaParts.push(String(captainTypeLabel));
      }
      metaParts.push(game.is_won ? 'Win' : 'Loss');
      if (typeof game.f1_score === 'number') {
        metaParts.push(`F1 ${formatDecimal(game.f1_score)}`);
      }
      if (game.board_id !== undefined) {
        metaParts.push(`Board ${game.board_id}`);
      }
      meta.textContent = metaParts.join(' • ');

      button.appendChild(title);
      button.appendChild(meta);

      button.addEventListener('click', () => {
        setGame(gameIndex);
      });

      elements.gameList.appendChild(button);
      gameButtons.set(gameIndex, button);
    });

    highlightActiveGameButton();
    updateGameNavButtons();
  }

  function highlightActiveGameButton() {
    if (!elements.gameList) return;
    let activeId = null;
    gameButtons.forEach((button, index) => {
      const isActive = index === state.currentGameIndex;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
      if (isActive) {
        activeId = button.id;
      }
    });
    if (activeId) {
      elements.gameList.setAttribute('aria-activedescendant', activeId);
    } else {
      elements.gameList.removeAttribute('aria-activedescendant');
    }
  }

  function clearViewer() {
    clearPlaybackTimer();
    playback.isPlaying = false;
    updatePlayButton();
    state.currentStage = 0;
    state.hasActiveGame = false;
    cancelSliderAnimation();
    disableSliderStepless();

    if (elements.metrics) {
      elements.metrics.innerHTML = '';
    }
    if (elements.board) {
      elements.board.innerHTML = '';
    }
    if (elements.boardCaption) {
      elements.boardCaption.textContent = 'Select a game to view the board.';
    }
    if (elements.shipTracker) {
      elements.shipTracker.innerHTML = '';
    }
    if (elements.eventDetails) {
      elements.eventDetails.innerHTML = '';
    }
    if (elements.progressDetails) {
      elements.progressDetails.innerHTML = '';
    }
    if (elements.timelineList) {
      elements.timelineList.innerHTML = '';
    }
    if (elements.frameLabel) {
      elements.frameLabel.textContent = '';
    }
    if (elements.eventLabel) {
      elements.eventLabel.textContent = '';
    }
    if (elements.slider) {
      setSliderValue(0);
      elements.slider.disabled = true;
      elements.slider.max = '0';
      elements.slider.setAttribute('aria-valuemax', '0');
      elements.slider.setAttribute('aria-valuenow', '0');
    }

    timelineButtons = [];
    updateGameNavButtons();
  }

  function applyFilters(options = {}) {
    const { preserveSelection = true } = options;
    const games = state.data?.games ?? [];
    const filtered = [];

    games.forEach((game, index) => {
      if (!game) return;
      const matchesLLM = !state.filters.llm || game.captain_llm === state.filters.llm;
      const matchesType = !state.filters.type || game.captain_type === state.filters.type;
      if (matchesLLM && matchesType) {
        filtered.push(index);
      }
    });

    filtered.sort((indexA, indexB) => {
      const gameA = games[indexA];
      const gameB = games[indexB];
      if (!gameA || !gameB) {
        return indexA - indexB;
      }

      const llmComparison = compareByOrder(gameA.captain_llm, gameB.captain_llm, LLM_ORDER);
      if (llmComparison !== 0) {
        return llmComparison;
      }

      const typeComparison = compareByOrder(gameA.captain_type, gameB.captain_type, CAPTAIN_TYPE_ORDER);
      if (typeComparison !== 0) {
        return typeComparison;
      }

      return indexA - indexB;
    });

    state.filteredIndices = filtered;
    renderGameList();

    if (!filtered.length) {
      setStatusMessage('No games match the selected filters. Try a different combination.', true);
      clearViewer();
      return;
    }

    const preserveCurrent = preserveSelection && state.hasActiveGame && filtered.includes(state.currentGameIndex);
    if (preserveCurrent) {
      const game = getCurrentGame();
      if (game) {
        updateStatusForGame(game);
        highlightActiveGameButton();
        updateGameNavButtons();
      }
    } else {
      setGame(filtered[0]);
    }
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
    cancelSliderAnimation();
    if (!playback.isPlaying) {
      disableSliderStepless();
    }
    const totalEvents = Array.isArray(game.events) ? game.events.length : 0;
    const maxStage = Math.max(totalEvents - 1, 0);
    elements.slider.max = String(maxStage);
    elements.slider.setAttribute('aria-valuemax', String(maxStage));
    setSliderValue(state.currentStage);
    elements.slider.setAttribute('aria-valuenow', String(state.currentStage));
    elements.slider.disabled = maxStage === 0;
  }

  function buildTimeline(game) {
    if (!elements.timelineList) return;
    timelineButtons = [];
    elements.timelineList.innerHTML = '';
    if (!Array.isArray(game.events)) return;

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
        setStage(index, { scrollTimeline: true });
      });

      elements.timelineList.appendChild(button);
      timelineButtons.push(button);
    });
  }

  function highlightActiveTimeline() {
    let targetButton = null;
    timelineButtons.forEach((button, index) => {
      const isActive = index === state.currentStage;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      if (isActive) {
        targetButton = button;
      }
    });
    if (timelineScrollRequested && targetButton) {
      requestAnimationFrame(() => {
        targetButton.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      });
    }
    timelineScrollRequested = false;
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

    // Build tracker rows: unknown color until sunk
    const rows = getShipTrackerRows(game.true_board, partialBoard);
    if (!rows.length) {
      elements.shipTracker.textContent = 'No ships detected on this board.';
      return;
    }

    const neutral = getColor(0); // water gray used for unknown/unsunk ships

    rows.forEach((rowInfo) => {
      const row = document.createElement('div');
      row.className = 'ship-row';

      const info = document.createElement('div');
      info.className = 'ship-info';

      const icon = document.createElement('i');
      icon.className = 'fas fa-ship ship-icon';
      icon.setAttribute('aria-hidden', 'true');

      const label = document.createElement('span');
      label.className = 'ship-label';
      label.textContent = `Length ${rowInfo.length}`;

      const segments = document.createElement('div');
      segments.className = 'ship-segments';

      const shipColor = rowInfo.sunkSymbol
        ? getColor(shipIdBySymbol[rowInfo.sunkSymbol] || rowInfo.id)
        : neutral;

      icon.style.color = shipColor;

      for (let k = 0; k < rowInfo.length; k += 1) {
        const seg = document.createElement('span');
        seg.className = 'ship-segment';
        seg.style.background = shipColor;
        seg.style.border = '1px solid #ffffff';
        seg.setAttribute('aria-hidden', 'true');
        segments.appendChild(seg);
      }

      const status = document.createElement('span');
      status.className = 'ship-status';
      status.textContent = 'Sunk ✓';
      status.classList.toggle('is-hidden', !rowInfo.sunkSymbol);

      info.appendChild(icon);
      info.appendChild(label);
      info.appendChild(segments);

      row.appendChild(info);
      row.appendChild(status);

      elements.shipTracker.appendChild(row);
    });
  }

  function renderEventDetails(game, event) {
    if (!elements.eventDetails) return;

    const container = elements.eventDetails;
    container.innerHTML = '';
    const decision = event?.decision ?? 'event';
    container.dataset.eventDecision = decision;

    activeEventAnimationId += 1;
    const animationId = activeEventAnimationId;

    const log = document.createElement('div');
    log.className = 'event-chat-log';
    container.appendChild(log);

    if (!event) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = 'No event details for this turn.';
      log.appendChild(empty);
      return;
    }

    const animationPromise = (async () => {
      if (decision === 'question') {
        const questionText = event.question?.text?.trim() || 'The captain posed a question.';
        await displayTypewrittenMessage({
          logEl: log,
          role: 'captain',
          label: 'Captain',
          text: questionText,
          animationId,
        });
        if (animationId !== activeEventAnimationId) return;

        const spotterMessage = createChatMessage({ role: 'spotter', label: 'Spotter' });
        log.appendChild(spotterMessage.wrapper);
        log.scrollTop = log.scrollHeight;
        spotterMessage.textEl.textContent = '';
        spotterMessage.textEl.style.display = 'none';
        setTypingCaretVisibility(spotterMessage.caret, false);
        const thinkingEl = createThinkingElement();
        spotterMessage.bubble.insertBefore(thinkingEl, spotterMessage.textEl);

        await delayWithCancel(THINKING_BASE_DELAY_MS, animationId);
        if (animationId !== activeEventAnimationId) return;

        if (spotterMessage.bubble.contains(thinkingEl)) {
          spotterMessage.bubble.removeChild(thinkingEl);
        }
        spotterMessage.textEl.style.display = '';
        await typeText({
          textEl: spotterMessage.textEl,
          caret: spotterMessage.caret,
          text: formatAnswerText(event.answer),
          animationId,
          logEl: log,
        });
        return;
      }

      if (decision === 'move') {
        const coords = event.move?.coords;
        const tile = event.move?.tile || coordsToTile(coords) || 'target';
        await displayTypewrittenMessage({
          logEl: log,
          role: 'captain',
          label: 'Captain',
          text: `Fire at ${tile}`,
          animationId,
        });
        if (animationId !== activeEventAnimationId) return;

        await delayWithCancel(RESULT_REVEAL_DELAY_MS, animationId);
        if (animationId !== activeEventAnimationId) return;

        const shipValue = Array.isArray(coords) ? game.true_board?.[coords[0]]?.[coords[1]] : null;
        const hit = typeof shipValue === 'number' && shipValue > 0 && event.board?.[coords[0]]?.[coords[1]] === shipValue;
        const resultText = hit ? `Hit ${getShipName(shipValue)}` : 'Missed';
        const resultEl = buildResultElement({ isHit: hit, text: resultText });
        log.appendChild(resultEl);
        log.scrollTop = log.scrollHeight;
        return;
      }

      const fallbackMessage = createChatMessage({ role: 'system', label: 'Update' });
      log.appendChild(fallbackMessage.wrapper);
      fallbackMessage.textEl.textContent = 'No additional details for this event.';
      setTypingCaretVisibility(fallbackMessage.caret, false);
    })();

    animationPromise.catch((error) => {
      console.error('Failed to animate event details', error);
    });

    return animationPromise;
  }

  function renderProgress(game, partialBoard) {
    if (!elements.progressDetails) return;
    const eventsSoFar = game.events.slice(0, state.currentStage + 1);
    const movesSoFar = eventsSoFar.filter((event) => event.decision === 'move').length;
    const questionsSoFar = eventsSoFar.filter((event) => event.decision === 'question').length;
    const hitsSoFar = countCells(partialBoard, (value) => typeof value === 'number' && value > 0);
    const missesSoFar = countCells(partialBoard, (value) => value === 0);

    const progressEntries = [
      { label: 'Moves', value: `${Math.min(movesSoFar, 40)}/40` },
      { label: 'Questions', value: `${Math.min(questionsSoFar, 15)}/15` },
      { label: 'Hits', value: `${hitsSoFar}` },
      { label: 'Misses', value: `${missesSoFar}` },
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
        return `Shot ${tile} → Hit ${getShipName(shipValue)}`;
      }
      return `Shot ${tile} → Miss`;
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
    if (!game || !event) {
      const fallback = Promise.resolve();
      lastRenderedStageIndex = state.currentStage;
      lastEventAnimationPromise = fallback;
      return fallback;
    }
    renderBoard(game, event);
    renderShipTracker(game, event.board);
    const animationPromise = renderEventDetails(game, event);
    lastRenderedStageIndex = state.currentStage;
    lastEventAnimationPromise = Promise.resolve(animationPromise);
    renderProgress(game, event.board);
    updateFrameAndEventLabels(game, event);
    highlightActiveTimeline();
    if (elements.slider) {
      elements.slider.setAttribute('aria-valuenow', String(state.currentStage));
    }
    return lastEventAnimationPromise;
  }

  function setGame(index) {
    const games = state.data?.games;
    if (!Array.isArray(games) || games.length === 0) return;
    const clampedIndex = Math.max(0, Math.min(index, games.length - 1));
    state.currentGameIndex = clampedIndex;
    state.currentStage = 0;
    const game = getCurrentGame();
    if (!game) return;
    const wasActive = state.hasActiveGame;
    state.hasActiveGame = true;
    updateStatusForGame(game);
    renderMetrics(game);
    buildTimeline(game);
    updateSlider(game);
    timelineScrollRequested = wasActive;
    highlightActiveGameButton();
    updateGameNavButtons();
    setPlaying(false);
    renderStageDependent();
  }

  function setStage(stage, options = {}) {
    const game = getCurrentGame();
    if (!game || !Array.isArray(game.events) || game.events.length === 0) return;
    const clampedStage = Math.max(0, Math.min(stage, game.events.length - 1));
    const prevStage = state.currentStage;
    const {
      scrollTimeline = true,
      animateSlider = false,
      animationDuration,
    } = options;
    const effectiveDuration = typeof animationDuration === 'number'
      ? animationDuration
      : playback.intervalMs;
    timelineScrollRequested = scrollTimeline && clampedStage !== prevStage;
    state.currentStage = clampedStage;
    if (elements.slider) {
      if (animateSlider && effectiveDuration > 0 && clampedStage !== prevStage) {
        enableSliderStepless();
        animateSliderValue(prevStage, clampedStage, effectiveDuration, () => {
          if (!playback.isPlaying) {
            disableSliderStepless();
          }
        });
      } else {
        cancelSliderAnimation();
        setSliderValue(clampedStage);
        if (!playback.isPlaying) {
          disableSliderStepless();
        }
      }
      elements.slider.setAttribute('aria-valuenow', String(clampedStage));
    }
    const animationPromise = renderStageDependent();
    if (playback.isPlaying) {
      schedulePlaybackAfterEvent(animationPromise, game, clampedStage);
    } else {
      clearPlaybackTimer();
    }
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
    if (elements.filterLLM) {
      elements.filterLLM.addEventListener('change', (event) => {
        state.filters.llm = event.target.value || '';
        applyFilters();
      });
    }

    if (elements.filterType) {
      elements.filterType.addEventListener('change', (event) => {
        state.filters.type = event.target.value || '';
        applyFilters();
      });
    }

    if (elements.slider) {
      elements.slider.addEventListener('input', (event) => {
        const rawValue = Number(event.target.value);
        const stage = Number.isNaN(rawValue) ? 0 : Math.round(rawValue);
        if (!Number.isNaN(rawValue) && rawValue !== stage) {
          event.target.value = String(stage);
        }
        setStage(stage, { scrollTimeline: false });
      });
    }

    if (elements.viewCaptain) {
      elements.viewCaptain.addEventListener('click', () => setView('captain'));
    }

    if (elements.viewSpotter) {
      elements.viewSpotter.addEventListener('click', () => setView('spotter'));
    }

    if (elements.playButton) {
      elements.playButton.addEventListener('click', () => {
        setPlaying(!playback.isPlaying);
      });
    }

    if (elements.timelineReset) {
      elements.timelineReset.addEventListener('click', () => {
        const game = getCurrentGame();
        if (!game || !Array.isArray(game.events) || !game.events.length) return;
        setStage(0);
      });
    }

    if (elements.timelineSkip) {
      elements.timelineSkip.addEventListener('click', () => {
        const game = getCurrentGame();
        if (!game || !Array.isArray(game.events) || !game.events.length) return;
        setStage(game.events.length - 1);
        setPlaying(false);
      });
    }

    if (elements.prevGame) {
      elements.prevGame.addEventListener('click', () => {
        changeGame(-1);
      });
    }

    if (elements.nextGame) {
      elements.nextGame.addEventListener('click', () => {
        changeGame(1);
      });
    }

    if (elements.speedSelect) {
      elements.speedSelect.addEventListener('change', (event) => {
        const value = Number(event.target.value);
        if (!Number.isFinite(value) || value <= 0) return;
        setPlaybackSpeed(value);
      });
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
      populateFilterOptions(data.games);
      attachEventListeners();
      updateViewButtons();
      updatePlayButton();
      updateSpeedControl();
      applyFilters({ preserveSelection: false });
    } catch (error) {
      console.error('Failed to load trajectory samples', error);
      setStatusMessage('Unable to load trajectory samples. Please refresh the page.', true);
    }
  }

  function initMotivationChat() {
    const root = document.getElementById('motivation-chat');
    const logEl = root ? root.querySelector('.event-chat-log') : null;
    const statusEl = document.getElementById('motivation-chat-status');
    const replayButton = document.getElementById('motivation-chat-replay');
    const abstractToggleButton = document.getElementById('motivation-show-abstract');
    const abstractCloseButton = document.getElementById('motivation-hide-abstract');
    const abstractPanel = document.getElementById('motivation-abstract-panel');
    const abstractHeading = document.getElementById('abstract-heading');

    if (!root || !logEl || !statusEl) {
      return;
    }

  const reduceMotion = prefersReducedMotion();

    const conversation = [
      {
        id: 'reader-why',
        role: 'captain',
        label: 'Reader',
        text: 'OK, so why did you do this research?',
        delayAfter: 360,
        iconClass: 'fas fa-book-reader',
      },
      {
        id: 'authors-why',
        role: 'spotter',
        label: 'Authors',
        text: 'In many real-world settings like scientific discovery and medical diagnosis, AI agents need to form hypotheses and run targeted experiments. We wanted to understand whether agents act rationally in these settings when given limited resources, and how their behavior compares to humans.',
        showThinking: true,
        delayAfter: 480,
        iconClass: 'fa-solid fa-chalkboard-user fas fa-chalkboard-teacher',
      },
      {
        id: 'reader-how',
        role: 'captain',
        label: 'Reader',
        text: 'Sounds interesting! How did you study this?',
        delayAfter: 320,
        iconClass: 'fas fa-book-reader',
      },
      {
        id: 'authors-how',
        role: 'spotter',
        label: 'Authors',
        text: 'We introduce a decision-oriented dialogue task designed to evaluate agentic information-seeking. In Collaborative Battleship, players can ask natural language questions to reveal information about the positions of hidden ships on the board.',
        showThinking: true,
        delayAfter: 520,
        iconClass: 'fa-solid fa-chalkboard-user fas fa-chalkboard-teacher',
      },
      {
        id: 'reader-why-battleship',
        role: 'captain',
        label: 'Reader',
        text: "Why Battleship? Isn't that a kids' game?",
        delayAfter: 300,
        iconClass: 'fas fa-book-reader',
      },
      {
        id: 'authors-why-battleship',
        role: 'spotter',
        label: 'Authors',
        text: 'Battleship enables us to study behavioral dynamics in a controlled, minimal environment. Because we can compute information-theoretic values—like “how useful was this question?” or “how good was that move?”—we can rigorously compare the decision-making of both human and AI players.',
        showThinking: true,
        delayAfter: 480,
        iconClass: 'fa-solid fa-chalkboard-user fas fa-chalkboard-teacher',
      },
      {
        id: 'reader-findings',
        role: 'captain',
        label: 'Reader',
        text: 'Got it. So what did you find?',
        delayAfter: 320,
        iconClass: 'fas fa-book-reader',
      },
      {
        id: 'authors-findings',
        role: 'spotter',
        label: 'Authors',
        text: 'We found some key skill gaps -- compared to human players, language model agents struggle to ground answers in context, generate informative questions, and select high-value actions. To address these gaps, we developed novel Monte Carlo inference strategies for LMs based on principles from Bayesian Experimental Design (BED), which significantly improved agent performance.',
        showThinking: true,
        delayAfter: 520,
        iconClass: 'fa-solid fa-chalkboard-user fas fa-chalkboard-teacher',
      },
      {
        id: 'reader-example',
        role: 'captain',
        label: 'Reader',
        text: 'That\'s impressive! Can you give an example?',
        delayAfter: 320,
        iconClass: 'fas fa-book-reader',
      },
      {
        id: 'authors-example',
        role: 'spotter',
        label: 'Authors',
        text: 'Sure! For example, we find that Llama-4-Scout is barely above random chance. However, with our methods, Llama-4-Scout improves dramatically at both asking and answering questions -- we see win rates of 82% against humans and 67% against GPT-5. This is especially exciting given that Llama-4-Scout costs about 100x less to run than GPT-5; our findings suggest that there\'s a lot of room to build more efficient AI systems.',
        showThinking: true,
        delayAfter: 520,
        iconClass: 'fa-solid fa-chalkboard-user fas fa-chalkboard-teacher',
      },
      {
        id: 'reader-generalize',
        role: 'captain',
        label: 'Reader',
        text: 'That sounds significant... but does any of this generalize beyond Battleship?',
        delayAfter: 320,
        iconClass: 'fas fa-book-reader',
      },
      {
        id: 'authors-generalize',
        role: 'spotter',
        label: 'Authors',
        text: 'Yes! With a few caveats described in the paper, our methods are general to many kinds of decision-making settings with partial information. As a first step, we replicated our findings on the “Guess Who?” game, where our methods significantly boosted accuracy by 28.3 to 42.4 percentage points. This demonstrates that our approach is broadly applicable for building rational information-seeking agents across different tasks.',
        showThinking: true,
        delayAfter: 520,
        iconClass: 'fa-solid fa-chalkboard-user fas fa-chalkboard-teacher',
      },
      {
        id: 'reader-learn-more',
        role: 'captain',
        label: 'Reader',
        text: 'That\'s great to hear! How can I learn more about this work?',
        delayAfter: 320,
        iconClass: 'fas fa-book-reader',
      },
      {
        id: 'authors-learn-more',
        role: 'spotter',
        label: 'Authors',
        text: 'You can check out our paper, which is available on arXiv. Also, keep scrolling below to check out more resources, including an interactive demo!',
        showThinking: true,
        iconClass: 'fa-solid fa-chalkboard-user fas fa-chalkboard-teacher',
      },
    ];

    const scheduledTimeouts = new Set();
    let activeAnimationToken = 0;
    let isAnimating = false;
    let hasStarted = false;
    let isAbstractExpanded = false;
  let abstractVisibilityFrame = null;
  let abstractHideListener = null;

    function scheduleTimeout(callback, duration) {
      const handle = window.setTimeout(() => {
        scheduledTimeouts.delete(handle);
        callback();
      }, Math.max(0, duration));
      scheduledTimeouts.add(handle);
      return handle;
    }

    function clearScheduledTimeouts() {
      scheduledTimeouts.forEach((handle) => {
        window.clearTimeout(handle);
      });
      scheduledTimeouts.clear();
    }

    function setStatus(stateClass, text) {
      statusEl.textContent = text;
      statusEl.classList.remove('is-idle', 'is-complete', 'is-error');
      if (stateClass) {
        statusEl.classList.add(stateClass);
      }
    }

    function setReplayDisabled(disabled) {
      if (!replayButton) return;
      if (disabled) {
        replayButton.disabled = true;
        replayButton.setAttribute('aria-disabled', 'true');
      } else {
        replayButton.disabled = false;
        replayButton.setAttribute('aria-disabled', 'false');
      }
    }

    function resetLog() {
      logEl.innerHTML = '';
    }

    function scrollLogToEnd() {
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setAbstractVisibility(expand, { focus = 'auto', announce = true } = {}) {
      const nextState = Boolean(expand);
      const previousState = isAbstractExpanded;
      isAbstractExpanded = nextState;

      if (abstractPanel) {
        if (abstractVisibilityFrame !== null) {
          window.cancelAnimationFrame(abstractVisibilityFrame);
          abstractVisibilityFrame = null;
        }

        if (abstractHideListener) {
          abstractPanel.removeEventListener('transitionend', abstractHideListener);
          abstractHideListener = null;
        }

        if (reduceMotion) {
          abstractPanel.hidden = !isAbstractExpanded;
          abstractPanel.classList.toggle('is-visible', isAbstractExpanded);
        } else if (previousState !== isAbstractExpanded) {
          if (isAbstractExpanded) {
            abstractPanel.hidden = false;
            abstractPanel.classList.remove('is-visible');
            abstractPanel.getBoundingClientRect();
            abstractVisibilityFrame = window.requestAnimationFrame(() => {
              abstractVisibilityFrame = null;
              if (!isAbstractExpanded) {
                if (!abstractPanel.hidden) {
                  abstractPanel.hidden = true;
                }
                return;
              }
              abstractPanel.classList.add('is-visible');
            });
          } else {
            abstractPanel.classList.remove('is-visible');
            if (!abstractPanel.hidden) {
              abstractHideListener = (event) => {
                if (event.target !== abstractPanel || event.propertyName !== 'opacity') {
                  return;
                }
                if (!isAbstractExpanded) {
                  abstractPanel.hidden = true;
                }
                if (abstractHideListener) {
                  abstractPanel.removeEventListener('transitionend', abstractHideListener);
                  abstractHideListener = null;
                }
              };
              abstractPanel.addEventListener('transitionend', abstractHideListener);
            } else {
              abstractPanel.hidden = true;
            }
          }
        } else {
          abstractPanel.hidden = !isAbstractExpanded;
          abstractPanel.classList.toggle('is-visible', isAbstractExpanded);
        }

        abstractPanel.setAttribute('aria-hidden', isAbstractExpanded ? 'false' : 'true');
      }

      if (abstractToggleButton) {
        abstractToggleButton.setAttribute('aria-expanded', isAbstractExpanded ? 'true' : 'false');
        const labelEl = abstractToggleButton.querySelector('.label');
        if (labelEl) {
          labelEl.textContent = isAbstractExpanded ? 'Hide full abstract' : 'View full abstract';
        }
      }

      if (abstractCloseButton) {
        abstractCloseButton.hidden = !isAbstractExpanded;
        abstractCloseButton.setAttribute('aria-hidden', isAbstractExpanded ? 'false' : 'true');
      }

      if (announce) {
        setStatus(null, '');
      }

      if (focus === 'none') {
        return;
      }

      if (isAbstractExpanded) {
        if (abstractHeading && (focus === 'auto' || focus === 'abstract')) {
          const delay = reduceMotion ? 0 : 120;
          window.setTimeout(() => {
            abstractHeading.focus({ preventScroll: false });
          }, delay);
        }
        return;
      }

      if (focus === 'toggle' && abstractToggleButton) {
        const delay = reduceMotion ? 0 : 120;
        window.setTimeout(() => {
          abstractToggleButton.focus({ preventScroll: false });
        }, delay);
        return;
      }

      if (focus === 'auto' || focus === 'log') {
        const delay = reduceMotion ? 0 : 120;
        const prevTabIndex = logEl.getAttribute('tabindex');
        if (prevTabIndex === null) {
          logEl.setAttribute('tabindex', '-1');
        }
        window.setTimeout(() => {
          logEl.focus({ preventScroll: false });
          if (prevTabIndex === null) {
            logEl.removeAttribute('tabindex');
          }
        }, delay);
      }
    }

    function typeTextWithToken({ textEl, caret, text, animationToken, logElement, timeScale = 1 }) {
      return new Promise((resolve) => {
        const messageText = typeof text === 'string' ? text : '';
        if (!textEl) {
          resolve();
          return;
        }

        if (reduceMotion || animationToken !== activeAnimationToken || messageText.length === 0) {
          textEl.textContent = messageText;
          setTypingCaretVisibility(caret, false);
          if (logElement) {
            logElement.scrollTop = logElement.scrollHeight;
          }
          resolve();
          return;
        }

        textEl.textContent = '';
        setTypingCaretVisibility(caret, true);
        let index = 0;

        const step = () => {
          if (animationToken !== activeAnimationToken) {
            textEl.textContent = messageText;
            setTypingCaretVisibility(caret, false);
            if (logElement) {
              logElement.scrollTop = logElement.scrollHeight;
            }
            resolve();
            return;
          }

          textEl.textContent += messageText.charAt(index);
          index += 1;

          if (logElement) {
            logElement.scrollTop = logElement.scrollHeight;
          }

          if (index >= messageText.length) {
            setTypingCaretVisibility(caret, false);
            resolve();
            return;
          }

          const previousChar = messageText.charAt(index - 1);
          let delay = TYPEWRITER_CHAR_DELAY_MS;
          if (/[,.;!?]/.test(previousChar)) {
            delay = TYPEWRITER_PUNCTUATION_DELAY_MS;
          } else if (previousChar === ' ') {
            delay = TYPEWRITER_SPACE_DELAY_MS;
          }

          scheduleTimeout(step, delay * timeScale);
        };

        step();
      });
    }

    function waitWithToken(durationMs, animationToken) {
      if (reduceMotion || durationMs <= 0) {
        return Promise.resolve();
      }
      return new Promise((resolve) => {
        scheduleTimeout(() => {
          if (animationToken !== activeAnimationToken) {
            resolve();
            return;
          }
          resolve();
        }, durationMs);
      });
    }

    async function renderMessage(entry, animationToken) {
      const message = createChatMessage({ role: entry.role, label: entry.label, iconClass: entry.iconClass });
      logEl.appendChild(message.wrapper);
      scrollLogToEnd();

      if (entry.showThinking && !reduceMotion) {
        const thinking = createThinkingElement();
        message.textEl.style.display = 'none';
        setTypingCaretVisibility(message.caret, false);
        message.bubble.insertBefore(thinking, message.textEl);
        const thinkingDelay = THINKING_BASE_DELAY_MS + (entry.thinkingDelay || 0);
        await waitWithToken(thinkingDelay, animationToken);
        if (animationToken !== activeAnimationToken) {
          return;
        }
        if (message.bubble.contains(thinking)) {
          message.bubble.removeChild(thinking);
        }
        message.textEl.style.display = '';
      }

      await typeTextWithToken({
        textEl: message.textEl,
        caret: message.caret,
        text: entry.text,
        animationToken,
        logElement: logEl,
      });
    }

    async function playConversation() {
      if (isAnimating) {
        return;
      }

      if (isAbstractExpanded) {
        setAbstractVisibility(false, { focus: 'none', announce: false });
      }

      isAnimating = true;
      hasStarted = true;
      setReplayDisabled(true);
      clearScheduledTimeouts();
      activeAnimationToken += 1;
      const animationToken = activeAnimationToken;
      resetLog();

      try {
        for (let i = 0; i < conversation.length; i += 1) {
          const entry = conversation[i];
          if (entry.delayBefore) {
            await waitWithToken(entry.delayBefore, animationToken);
            if (animationToken !== activeAnimationToken) {
              return;
            }
          }
          await renderMessage(entry, animationToken);
          if (animationToken !== activeAnimationToken) {
            return;
          }
          if (entry.delayAfter) {
            await waitWithToken(entry.delayAfter, animationToken);
            if (animationToken !== activeAnimationToken) {
              return;
            }
          }
        }
      } finally {
        clearScheduledTimeouts();
        if (animationToken === activeAnimationToken) {
          isAnimating = false;
          setReplayDisabled(false);
        }
      }
    }

    function startPlayback() {
      if (reduceMotion || isAnimating) {
        return;
      }
      playConversation();
    }

    function expandAbstract({ focus = 'abstract', announce = true } = {}) {
      setAbstractVisibility(true, { focus, announce });
    }

    function collapseAbstract({ focus = 'log', announce = true } = {}) {
      if (!isAbstractExpanded) {
        if (announce || focus !== 'none') {
          setAbstractVisibility(false, { focus, announce });
        }
        return;
      }
      setAbstractVisibility(false, { focus, announce });
    }

    function initialiseTranscript() {
      resetLog();
      conversation.forEach((entry) => {
        const message = createChatMessage({ role: entry.role, label: entry.label, iconClass: entry.iconClass });
        message.textEl.textContent = entry.text;
        setTypingCaretVisibility(message.caret, false);
        logEl.appendChild(message.wrapper);
      });
      scrollLogToEnd();
    }

    if (reduceMotion) {
      initialiseTranscript();
      setStatus(null, '');
      setReplayDisabled(true);
    } else {
      setReplayDisabled(false);
      setStatus(null, '');

      if (replayButton) {
        replayButton.addEventListener('click', (event) => {
          event.preventDefault();
          if (isAbstractExpanded) {
            collapseAbstract({ focus: 'none', announce: false });
          }
          startPlayback();
        });
      }

      if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries, obs) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting && !hasStarted && !isAnimating && !isAbstractExpanded) {
              startPlayback();
              if (obs) {
                obs.unobserve(root);
              }
            }
          });
        }, { threshold: 0.35 });
        observer.observe(root);
      } else {
        scheduleTimeout(() => {
          if (!hasStarted && !isAnimating && !isAbstractExpanded) {
            startPlayback();
          }
        }, 600);
      }
    }

    if (abstractToggleButton) {
      abstractToggleButton.addEventListener('click', (event) => {
        event.preventDefault();
        const nextState = !isAbstractExpanded;
        if (nextState) {
          expandAbstract({ focus: 'abstract' });
        } else {
          collapseAbstract({ focus: 'toggle' });
        }
      });
    }

    if (abstractCloseButton) {
      abstractCloseButton.addEventListener('click', (event) => {
        event.preventDefault();
        collapseAbstract({ focus: 'toggle' });
      });
    }

    setAbstractVisibility(false, { focus: 'none', announce: false });

    if (window.location.hash === '#abstract') {
      expandAbstract({ focus: 'abstract', announce: false });
    }

    window.addEventListener('hashchange', () => {
      if (window.location.hash === '#abstract') {
        expandAbstract({ focus: 'abstract', announce: false });
      } else if (isAbstractExpanded) {
        collapseAbstract({ focus: 'none', announce: false });
      }
    });

    if (reduceMotion) {
      setReplayDisabled(true);
    }
  }

  updateSpeedControl();
  updateGameNavButtons();
  initMotivationChat();
  loadData();
});
