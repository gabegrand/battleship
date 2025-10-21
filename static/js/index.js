const BASE_COLOR_PALETTE_ENTRIES = [
  [-1, '#eaeae4'],
  [0, '#9b9c97'],
  [1, '#ac2028'],
  [2, '#04af70'],
  [3, '#6d467b'],
  [4, '#ffa500'],
];

const FALLBACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#2b908f', '#f45b5b', '#91e8e1', '#6f38a0'];

const SHIP_NAMES = { 1: 'Red ship', 2: 'Green ship', 3: 'Purple ship', 4: 'Orange ship' };
const SHIP_SYMBOLS = { 1: 'R', 2: 'G', 3: 'P', 4: 'O' };
const SHIP_ID_BY_SYMBOL = Object.fromEntries(Object.entries(SHIP_SYMBOLS).map(([id, sym]) => [sym, Number(id)]));

const DEFAULT_PLAYBACK_INTERVAL_MS = 1000;

const LLM_ORDER = ['GPT-5', 'GPT-4o', 'Llama-4-Scout', 'Baseline'];

const LLM_LABELS = {
  'GPT-5': 'GPT-5',
  'GPT-4o': 'GPT-4o',
  'Llama-4-Scout': 'Llama-4-Scout',
  Baseline: 'Baseline (No LLM)',
};

const CAPTAIN_TYPE_ORDER = ['LM', '+Bayes-Q', '+Bayes-M', '+Bayes-QM', '+Bayes-QMD', 'Random', 'Greedy'];

const CAPTAIN_TYPE_LABELS = {
  LM: 'LM-only',
  '+Bayes-Q': 'LM+Bayes-Q',
  '+Bayes-M': 'LM+Bayes-M',
  '+Bayes-QM': 'LM+Bayes-QM',
  '+Bayes-QMD': 'LM+Bayes-QMD',
  Random: 'Random',
  Greedy: 'Greedy',
};

const ICON_CLASSES = {
  captain: 'fas fa-anchor',
  spotter: 'fas fa-binoculars',
  miss: 'fas fa-times',
  hit: 'fas fa-bullseye',
};

const TYPEWRITER_CHAR_DELAY_MS = 26;
const TYPEWRITER_PUNCTUATION_DELAY_MS = 140;
const TYPEWRITER_SPACE_DELAY_MS = 35;
const THINKING_BASE_DELAY_MS = 520;
const RESULT_REVEAL_DELAY_MS = 220;
const QUESTION_LINGER_BUFFER_MS = 900;
const MOVE_LINGER_BUFFER_MS = 320;
const OTHER_LINGER_BUFFER_MS = 500;
const SLIDER_ANIMATION_DURATION_RATIO = 1.0;

const DEFAULT_SELECTORS = {
  status: '#trajectory-status',
  filterLLM: '#filter-llm',
  filterType: '#filter-type',
  gameList: '#trajectory-game-list',
  viewCaptain: '#view-captain',
  viewSpotter: '#view-spotter',
  metrics: '#trajectory-metrics',
  board: '#trajectory-board',
  boardCaption: '#board-caption',
  slider: '#trajectory-slider',
  frameLabel: '#frame-label',
  eventLabel: '#event-label',
  eventDetails: '#event-details',
  progressDetails: '#progress-details',
  shipTracker: '#ship-tracker',
  timelineList: '#trajectory-event-list',
  playButton: '#timeline-play',
  timelineReset: '#timeline-reset',
  timelineSkip: '#timeline-skip',
  prevGame: '#prev-game',
  nextGame: '#next-game',
  speedSelect: '#playback-speed',
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

function createChatMessage({ role, label, iconClass: explicitIconClass }) {
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
    const resolvedIconClass = explicitIconClass || getAvatarIconClass(role);
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

function truncate(text, maxLength = 70) {
  if (!text) return '';
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function formatDecimal(value, digits = 2) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '–';
  }
  return value.toFixed(digits);
}

function coordsToTile(coords) {
  if (!Array.isArray(coords) || coords.length < 2) {
    return '';
  }
  const [row, col] = coords;
  return `${String.fromCharCode(65 + row)}${col + 1}`;
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

function getShipTrackerRows(trueBoard, partialBoard) {
  if (!Array.isArray(trueBoard) || !Array.isArray(partialBoard)) return [];
  let maxId = 0;
  trueBoard.forEach((row) => {
    if (!Array.isArray(row)) return;
    row.forEach((value) => {
      const numeric = Number(value);
      if (Number.isFinite(numeric) && numeric > 0) {
        maxId = Math.max(maxId, numeric);
      }
    });
  });

  const rows = [];
  for (let shipType = 1; shipType <= maxId; shipType += 1) {
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
      const symbol = SHIP_SYMBOLS[shipType] || String(shipType);
      const sunk = stateCount === targetCount;
      rows.push({ length: targetCount, sunkSymbol: sunk ? symbol : null, id: shipType });
    }
  }

  rows.sort((a, b) => b.length - a.length);
  return rows;
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

function easeOutCubic(t) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const delta = 1 - t;
  return 1 - delta * delta * delta;
}

function computeIntervalForSpeed(speed) {
  const multiplier = Number(speed);
  if (!Number.isFinite(multiplier) || multiplier <= 0) {
    return DEFAULT_PLAYBACK_INTERVAL_MS;
  }
  return DEFAULT_PLAYBACK_INTERVAL_MS / multiplier;
}

let trajectoryDataPromise = null;

function getTrajectoryData() {
  if (!trajectoryDataPromise) {
    trajectoryDataPromise = fetch('./static/data/trajectory_samples.json', { cache: 'no-store' })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!data || !Array.isArray(data.games) || data.games.length === 0) {
          throw new Error('No trajectory samples found.');
        }
        return data;
      });
  }
  return trajectoryDataPromise;
}

class TrajectoryExplorer {
  constructor(root, options = {}) {
    this.root = root;
    this.config = Object.assign(
      {
        selectors: DEFAULT_SELECTORS,
        initialView: 'captain',
        playbackSpeed: 1,
        autoplay: false,
        autoAdvanceGames: false,
        loopGames: false,
        dataPromise: getTrajectoryData(),
      },
      options,
    );

    this.elements = this.resolveElements(this.config.selectors);
    this.colorPalette = new Map(BASE_COLOR_PALETTE_ENTRIES);
    this.fallbackIndex = 0;
    this.gameButtons = new Map();
    this.timelineButtons = [];
    this.timelineScrollRequested = false;

    this.state = {
      data: null,
      currentGameIndex: 0,
      currentStage: 0,
      currentView: this.config.initialView,
      filters: {
        llm: '',
        type: '',
      },
      filteredIndices: [],
      hasActiveGame: false,
      playbackSpeed: Number(this.config.playbackSpeed) || 1,
    };

    this.playback = {
      intervalMs: computeIntervalForSpeed(this.state.playbackSpeed),
      timerId: null,
      isPlaying: false,
      sequenceId: 0,
    };

    this.sliderAnimation = {
      frameId: null,
      startValue: 0,
      targetValue: 0,
      startTime: 0,
      duration: 0,
      onComplete: null,
    };

    this.sliderStepControl = {
      defaultStep: this.elements.slider ? this.elements.slider.getAttribute('step') || '1' : '1',
      isStepless: false,
    };

    this.activeEventAnimationId = 0;
    this.lastEventAnimationPromise = Promise.resolve();
    this.lastRenderedStageIndex = 0;

    this.setStatusMessage('Loading curated games…');
    this.updateSpeedControl();
    this.updateGameNavButtons();
    this.updatePlayButton();
    this.updateViewButtons();

    Promise.resolve(this.config.dataPromise || getTrajectoryData())
      .then((data) => this.initialize(data))
      .catch((error) => this.handleDataError(error));
  }

  resolveElements(selectors = {}) {
    const resolved = {};
    const mergedSelectors = Object.assign({}, DEFAULT_SELECTORS, selectors);
    Object.entries(mergedSelectors).forEach(([key, selector]) => {
      if (selector && typeof selector === 'string') {
        resolved[key] = this.root.querySelector(selector);
      } else {
        resolved[key] = null;
      }
    });
    return resolved;
  }

  initialize(data) {
    try {
      if (!data || !Array.isArray(data.games) || data.games.length === 0) {
        this.setStatusMessage('No trajectory samples found.', true);
        return;
      }
      this.state.data = data;
      this.populateFilterOptions(data.games);
      this.attachEventListeners();
      this.updateViewButtons();
      this.updatePlayButton();
      this.updateSpeedControl();
      this.applyFilters({ preserveSelection: false });
      if (this.config.autoplay) {
        this.setPlaybackSpeed(this.state.playbackSpeed);
        this.setPlaying(true);
      }
    } catch (error) {
      this.handleDataError(error);
    }
  }

  handleDataError(error) {
    console.error('Failed to initialize trajectory explorer', error);
    this.setStatusMessage('Unable to load trajectory samples. Please refresh the page.', true);
  }

  setStatusMessage(message, isError = false) {
    if (!this.elements.status) return;
    this.elements.status.textContent = message;
    this.elements.status.classList.toggle('has-error', Boolean(isError));
  }

  updateStatusForGame(game) {
    if (!this.elements.status || !game) return;
    const parts = [
      `${game.captain_llm}`,
      `${game.captain_type}`,
      `Board ${game.board_id}`,
      `Round ${game.round_id}`,
    ];
    if (typeof game.seed === 'number') {
      parts.push(`Seed ${game.seed}`);
    }
    this.elements.status.textContent = parts.join(' • ');
    this.elements.status.classList.remove('has-error');
  }

  getColor(value) {
    const numeric = typeof value === 'number' ? value : Number(value);
    if (this.colorPalette.has(numeric)) {
      return this.colorPalette.get(numeric);
    }
    if (Number.isFinite(numeric) && numeric > 0) {
      const color = FALLBACK_COLORS[this.fallbackIndex % FALLBACK_COLORS.length];
      this.fallbackIndex += 1;
      this.colorPalette.set(numeric, color);
      return color;
    }
    return '#d0d4dc';
  }

  updatePlayButton() {
    if (!this.elements.playButton) return;
    const button = this.elements.playButton;
    const icon = button.querySelector('.playback-icon');
    const label = button.querySelector('.playback-text');
    if (this.playback.isPlaying) {
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

  clearPlaybackTimer() {
    this.playback.sequenceId += 1;
    if (this.playback.timerId !== null) {
      clearTimeout(this.playback.timerId);
      this.playback.timerId = null;
    }
  }

  cancelSliderAnimation() {
    if (this.sliderAnimation.frameId !== null) {
      window.cancelAnimationFrame(this.sliderAnimation.frameId);
      this.sliderAnimation.frameId = null;
    }
    this.sliderAnimation.onComplete = null;
  }

  getSliderMaxValue() {
    const slider = this.elements.slider;
    if (!slider) return 0;
    const maxAttr = Number(slider.max);
    if (Number.isFinite(maxAttr)) {
      return maxAttr;
    }
    const ariaMax = Number(slider.getAttribute('aria-valuemax'));
    return Number.isFinite(ariaMax) ? ariaMax : 0;
  }

  updateSliderProgress(value) {
    const slider = this.elements.slider;
    if (!slider) return;
    const max = this.getSliderMaxValue();
    const numericValue = Number(value);
    let percent = 0;
    if (Number.isFinite(numericValue) && max > 0) {
      percent = Math.min(Math.max(numericValue / max, 0), 1) * 100;
    }
    slider.style.setProperty('--slider-progress', `${percent}%`);
  }

  setSliderValue(value) {
    const slider = this.elements.slider;
    if (!slider) return;
    const numericValue = Number(value);
    const safeValue = Number.isFinite(numericValue) ? numericValue : 0;
    slider.value = String(safeValue);
    this.updateSliderProgress(safeValue);
  }

  enableSliderStepless() {
    const slider = this.elements.slider;
    if (!slider || this.sliderStepControl.isStepless) return;
    slider.setAttribute('step', 'any');
    this.sliderStepControl.isStepless = true;
  }

  disableSliderStepless() {
    const slider = this.elements.slider;
    if (!slider) return;
    if (this.sliderStepControl.defaultStep) {
      slider.setAttribute('step', this.sliderStepControl.defaultStep);
    } else {
      slider.removeAttribute('step');
    }
    this.sliderStepControl.isStepless = false;
  }

  animateSliderValue(startValue, targetValue, duration, onComplete) {
    const slider = this.elements.slider;
    if (!slider) return;
    this.cancelSliderAnimation();
    let effectiveDuration = duration;
    if (prefersReducedMotion()) {
      effectiveDuration = 0;
    }
    this.sliderAnimation.startValue = startValue;
    this.sliderAnimation.targetValue = targetValue;
    this.sliderAnimation.duration = effectiveDuration;
    this.sliderAnimation.startTime = performance.now();
    this.sliderAnimation.onComplete = typeof onComplete === 'function' ? onComplete : null;

    if (effectiveDuration <= 0 || startValue === targetValue) {
      this.setSliderValue(targetValue);
      this.sliderAnimation.frameId = null;
      if (this.sliderAnimation.onComplete) {
        this.sliderAnimation.onComplete();
        this.sliderAnimation.onComplete = null;
      }
      return;
    }

    this.setSliderValue(startValue);

    const step = (now) => {
      const elapsed = now - this.sliderAnimation.startTime;
      const progress = Math.min(elapsed / this.sliderAnimation.duration, 1);
      const eased = easeOutCubic(progress);
      const value = this.sliderAnimation.startValue + (this.sliderAnimation.targetValue - this.sliderAnimation.startValue) * eased;
      this.setSliderValue(value);
      if (progress < 1) {
        this.sliderAnimation.frameId = window.requestAnimationFrame(step);
      } else {
        this.sliderAnimation.frameId = null;
        this.setSliderValue(targetValue);
        if (this.sliderAnimation.onComplete) {
          this.sliderAnimation.onComplete();
          this.sliderAnimation.onComplete = null;
        }
      }
    };

    this.sliderAnimation.frameId = window.requestAnimationFrame(step);
  }

  computeLingerDuration(event) {
    const base = this.playback.intervalMs;
    let buffer = OTHER_LINGER_BUFFER_MS;
    if (event?.decision === 'question') {
      buffer = QUESTION_LINGER_BUFFER_MS;
    } else if (event?.decision === 'move') {
      buffer = MOVE_LINGER_BUFFER_MS;
    }
    const speed = this.state.playbackSpeed || 1;
    const adjustedBuffer = prefersReducedMotion() ? 0 : buffer / speed;
    return Math.max(0, base + adjustedBuffer);
  }

  schedulePlaybackAfterEvent(animationPromise, game, stageIndex) {
    this.clearPlaybackTimer();
    if (!this.playback.isPlaying) {
      return;
    }
    if (!game || !Array.isArray(game.events) || game.events.length === 0) {
      return;
    }

    const isLastStage = stageIndex >= game.events.length - 1;

    const sequenceId = ++this.playback.sequenceId;
    Promise.resolve(animationPromise)
      .catch((error) => {
        console.error('Event animation failed', error);
      })
      .then(() => {
        if (!this.playback.isPlaying || this.playback.sequenceId !== sequenceId) {
          return;
        }
        if (this.state.currentStage !== stageIndex) {
          return;
        }

        const lingerDuration = this.computeLingerDuration(game.events[stageIndex]);
        const sliderDuration = this.playback.intervalMs * SLIDER_ANIMATION_DURATION_RATIO;

        this.playback.timerId = window.setTimeout(() => {
          if (!this.playback.isPlaying || this.playback.sequenceId !== sequenceId) {
            return;
          }
          if (this.state.currentStage !== stageIndex) {
            return;
          }

          if (isLastStage) {
            if (this.config.autoAdvanceGames) {
              if (!this.advanceToNextGame()) {
                this.playback.isPlaying = false;
                this.updatePlayButton();
                if (this.sliderAnimation.frameId === null) {
                  this.disableSliderStepless();
                }
              }
            } else {
              this.playback.isPlaying = false;
              this.updatePlayButton();
              if (this.sliderAnimation.frameId === null) {
                this.disableSliderStepless();
              }
            }
            return;
          }

          const nextStage = stageIndex + 1;
          this.setStage(nextStage, {
            scrollTimeline: false,
            animateSlider: Boolean(this.elements.slider),
            animationDuration: sliderDuration,
          });
        }, lingerDuration);
      });
  }

  setPlaybackSpeed(speed) {
    const multiplier = Number(speed);
    if (!Number.isFinite(multiplier) || multiplier <= 0) {
      return;
    }
    if (Math.abs(multiplier - this.state.playbackSpeed) < 1e-6) {
      this.updateSpeedControl();
      return;
    }
    this.state.playbackSpeed = multiplier;
    this.playback.intervalMs = computeIntervalForSpeed(multiplier);
    this.clearPlaybackTimer();
    this.updateSpeedControl();
    if (this.playback.isPlaying) {
      this.schedulePlaybackAfterEvent(this.lastEventAnimationPromise, this.getCurrentGame(), this.lastRenderedStageIndex);
    }
  }

  updateSpeedControl() {
    if (!this.elements.speedSelect) return;
    const value = String(this.state.playbackSpeed);
    if (this.elements.speedSelect.value !== value) {
      this.elements.speedSelect.value = value;
    }
  }

  setButtonAvailability(button, enabled) {
    if (!button) return;
    button.disabled = !enabled;
    button.setAttribute('aria-disabled', enabled ? 'false' : 'true');
  }

  updateGameNavButtons() {
    if (!this.elements.prevGame || !this.elements.nextGame) return;
    const position = this.state.filteredIndices.indexOf(this.state.currentGameIndex);
    const hasGames = position !== -1 && this.state.filteredIndices.length > 0;
    const canGoPrev = hasGames && position > 0;
    const canGoNext = hasGames && position < this.state.filteredIndices.length - 1;
    this.setButtonAvailability(this.elements.prevGame, canGoPrev);
    this.setButtonAvailability(this.elements.nextGame, canGoNext);
  }

  changeGame(offset) {
    if (!Number.isInteger(offset) || this.state.filteredIndices.length === 0) return;
    let position = this.state.filteredIndices.indexOf(this.state.currentGameIndex);
    if (position === -1) {
      position = 0;
    }
    const nextPosition = Math.min(
      Math.max(position + offset, 0),
      this.state.filteredIndices.length - 1,
    );
    if (nextPosition === position) return;
    const nextGameIndex = this.state.filteredIndices[nextPosition];
    this.setGame(nextGameIndex);
    const button = this.gameButtons.get(nextGameIndex);
    if (button) {
      const activeElement = document.activeElement;
      const shouldMoveFocus = activeElement && activeElement !== document.body && this.root.contains(activeElement);
      if (shouldMoveFocus) {
        button.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        try {
          button.focus({ preventScroll: true });
        } catch (error) {
          button.focus();
        }
      }
    }
  }

  setPlaying(isPlaying) {
    const next = Boolean(isPlaying);
    const wasPlaying = this.playback.isPlaying;
    if (next === wasPlaying) {
      if (next) {
        this.schedulePlaybackAfterEvent(this.lastEventAnimationPromise, this.getCurrentGame(), this.lastRenderedStageIndex);
      } else {
        this.clearPlaybackTimer();
        this.disableSliderStepless();
      }
      return;
    }

    this.cancelSliderAnimation();

    if (next) {
      this.enableSliderStepless();
      const game = this.getCurrentGame();
      if (game && Array.isArray(game.events) && game.events.length > 0 && this.state.currentStage >= game.events.length - 1) {
        this.state.currentStage = 0;
        this.setSliderValue(0);
        this.timelineScrollRequested = false;
        const animationPromise = this.renderStageDependent();
        if (animationPromise) {
          this.lastEventAnimationPromise = animationPromise;
        }
      }
    } else {
      this.clearPlaybackTimer();
      this.disableSliderStepless();
    }

    this.playback.isPlaying = next;
    this.updatePlayButton();
    if (this.playback.isPlaying) {
      this.schedulePlaybackAfterEvent(this.lastEventAnimationPromise, this.getCurrentGame(), this.lastRenderedStageIndex);
    }
  }

  getCurrentGame() {
    return this.state.data?.games?.[this.state.currentGameIndex] ?? null;
  }

  getCurrentEvent() {
    const game = this.getCurrentGame();
    if (!game || !Array.isArray(game.events)) return null;
    return game.events[this.state.currentStage] ?? null;
  }

  updateSlider(game) {
    const slider = this.elements.slider;
    if (!slider) return;
    this.cancelSliderAnimation();
    if (!this.playback.isPlaying) {
      this.disableSliderStepless();
    }
    const totalEvents = Array.isArray(game.events) ? game.events.length : 0;
    const maxStage = Math.max(totalEvents - 1, 0);
    slider.max = String(maxStage);
    slider.setAttribute('aria-valuemax', String(maxStage));
    this.setSliderValue(this.state.currentStage);
    slider.setAttribute('aria-valuenow', String(this.state.currentStage));
    slider.disabled = maxStage === 0;
  }

  renderBoard(game, event) {
    const boardEl = this.elements.board;
    if (!boardEl) return;
    const trueBoard = game.true_board;
    const partialBoard = event.board;
    if (!Array.isArray(trueBoard) || !Array.isArray(partialBoard)) {
      boardEl.innerHTML = '<p>Board unavailable.</p>';
      return;
    }
    const size = trueBoard.length;
    boardEl.innerHTML = '';
    boardEl.style.setProperty('--board-columns', String(size + 1));

    const topLeft = document.createElement('div');
    topLeft.className = 'board-label-cell';
    topLeft.setAttribute('role', 'presentation');
    boardEl.appendChild(topLeft);

    for (let col = 0; col < size; col += 1) {
      const header = document.createElement('div');
      header.className = 'board-label-cell';
      header.textContent = String(col + 1);
      header.setAttribute('role', 'columnheader');
      boardEl.appendChild(header);
    }

    const currentCoords = event.decision === 'move' ? event.move?.coords : null;

    for (let row = 0; row < size; row += 1) {
      const rowLabel = document.createElement('div');
      rowLabel.className = 'board-label-cell';
      rowLabel.textContent = String.fromCharCode(65 + row);
      rowLabel.setAttribute('role', 'rowheader');
      boardEl.appendChild(rowLabel);

      for (let col = 0; col < size; col += 1) {
        const cell = document.createElement('div');
        cell.className = 'board-cell';
        const partialValue = partialBoard[row]?.[col];
        const trueValue = trueBoard[row]?.[col];
        const isCurrent = Array.isArray(currentCoords) && currentCoords[0] === row && currentCoords[1] === col;

        const displayValue = this.state.currentView === 'captain' ? partialValue : trueValue;
        cell.style.setProperty('--cell-bg', this.getColor(displayValue));
        if (this.state.currentView === 'spotter' && partialValue === -1) {
          cell.setAttribute('data-hidden', 'true');
        } else {
          cell.removeAttribute('data-hidden');
        }

        if (this.state.currentView === 'captain') {
          if (partialValue === 0) {
            cell.textContent = '•';
          } else if (partialValue > 0) {
            cell.textContent = SHIP_SYMBOLS[partialValue] || String(partialValue);
          }
        } else if (trueValue > 0) {
          cell.textContent = SHIP_SYMBOLS[trueValue] || String(trueValue);
        }

        if (isCurrent) {
          cell.classList.add('is-current');
        }

        cell.setAttribute('aria-label', `${String.fromCharCode(65 + row)}${col + 1}`);
        boardEl.appendChild(cell);
      }
    }
  }

  renderShipTracker(game, partialBoard) {
    const shipTrackerEl = this.elements.shipTracker;
    if (!shipTrackerEl) return;
    shipTrackerEl.innerHTML = '';
    if (!Array.isArray(game.true_board)) {
      shipTrackerEl.textContent = 'Ship data unavailable.';
      return;
    }

    const rows = getShipTrackerRows(game.true_board, partialBoard);
    if (!rows.length) {
      shipTrackerEl.textContent = 'No ships detected on this board.';
      return;
    }

    const neutral = this.getColor(0);

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
        ? this.getColor(SHIP_ID_BY_SYMBOL[rowInfo.sunkSymbol] || rowInfo.id)
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

      shipTrackerEl.appendChild(row);
    });
  }

  async displayTypewrittenMessage({ logEl, role, label, text, animationId }) {
    if (!logEl) return null;
    const message = createChatMessage({ role, label });
    logEl.appendChild(message.wrapper);
    logEl.scrollTop = logEl.scrollHeight;
    await this.typeText({
      textEl: message.textEl,
      caret: message.caret,
      text,
      animationId,
      logEl,
    });
    return message;
  }

  delayWithCancel(durationMs, animationId, scale = 1) {
    const duration = prefersReducedMotion() ? 0 : Math.max(0, durationMs * scale);
    if (duration === 0) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const timeoutId = window.setTimeout(() => {
        resolve();
      }, duration);
      if (animationId !== this.activeEventAnimationId) {
        window.clearTimeout(timeoutId);
        resolve();
      }
    });
  }

  typeText({ textEl, caret, text, animationId, logEl, timeScale = 1 }) {
    return new Promise((resolve) => {
      const messageText = typeof text === 'string' ? text : '';
      if (!textEl) {
        resolve();
        return;
      }

      if (prefersReducedMotion() || animationId !== this.activeEventAnimationId || messageText.length === 0) {
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
        if (animationId !== this.activeEventAnimationId) {
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

  renderEventDetails(game, event) {
    const container = this.elements.eventDetails;
    if (!container) return Promise.resolve();

    container.innerHTML = '';
    const decision = event?.decision ?? 'event';
    container.dataset.eventDecision = decision;

    this.activeEventAnimationId += 1;
    const animationId = this.activeEventAnimationId;

    const log = document.createElement('div');
    log.className = 'event-chat-log';
    container.appendChild(log);

    if (!event) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = 'No event details for this turn.';
      log.appendChild(empty);
      return Promise.resolve();
    }

    const animationPromise = (async () => {
      if (decision === 'question') {
        const questionText = event.question?.text?.trim() || 'The captain posed a question.';
        await this.displayTypewrittenMessage({
          logEl: log,
          role: 'captain',
          label: 'Captain',
          text: questionText,
          animationId,
        });
        if (animationId !== this.activeEventAnimationId) return;

        const spotterMessage = createChatMessage({ role: 'spotter', label: 'Spotter' });
        log.appendChild(spotterMessage.wrapper);
        log.scrollTop = log.scrollHeight;
        spotterMessage.textEl.textContent = '';
        spotterMessage.textEl.style.display = 'none';
        setTypingCaretVisibility(spotterMessage.caret, false);
        const thinkingEl = createThinkingElement();
        spotterMessage.bubble.insertBefore(thinkingEl, spotterMessage.textEl);

        await this.delayWithCancel(THINKING_BASE_DELAY_MS, animationId);
        if (animationId !== this.activeEventAnimationId) return;

        if (spotterMessage.bubble.contains(thinkingEl)) {
          spotterMessage.bubble.removeChild(thinkingEl);
        }
        spotterMessage.textEl.style.display = '';
        await this.typeText({
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
        await this.displayTypewrittenMessage({
          logEl: log,
          role: 'captain',
          label: 'Captain',
          text: `Fire at ${tile}`,
          animationId,
        });
        if (animationId !== this.activeEventAnimationId) return;

        await this.delayWithCancel(RESULT_REVEAL_DELAY_MS, animationId);
        if (animationId !== this.activeEventAnimationId) return;

        const shipValue = Array.isArray(coords) ? game.true_board?.[coords[0]]?.[coords[1]] : null;
        const hit = typeof shipValue === 'number' && shipValue > 0 && event.board?.[coords[0]]?.[coords[1]] === shipValue;
        const resultText = hit ? `Hit ${SHIP_NAMES[shipValue] || `Ship ${shipValue}`}` : 'Missed';
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

  renderProgress(game, partialBoard) {
    const progressEl = this.elements.progressDetails;
    if (!progressEl) return;
    const eventsSoFar = game.events.slice(0, this.state.currentStage + 1);
    const movesSoFar = eventsSoFar.filter((evt) => evt.decision === 'move').length;
    const questionsSoFar = eventsSoFar.filter((evt) => evt.decision === 'question').length;
    const hitsSoFar = countCells(partialBoard, (value) => typeof value === 'number' && value > 0);
    const missesSoFar = countCells(partialBoard, (value) => value === 0);

    const progressEntries = [
      { label: 'Moves', value: `${Math.min(movesSoFar, 40)}/40` },
      { label: 'Questions', value: `${Math.min(questionsSoFar, 15)}/15` },
      { label: 'Hits', value: `${hitsSoFar}` },
      { label: 'Misses', value: `${missesSoFar}` },
    ];

    progressEl.innerHTML = '';
    const grid = document.createElement('div');
    grid.className = 'progress-grid';
    progressEntries.forEach((entry) => {
      const chip = document.createElement('div');
      chip.className = 'progress-chip';
      chip.textContent = `${entry.label}: ${entry.value}`;
      grid.appendChild(chip);
    });
    progressEl.appendChild(grid);
  }

  createTimelineText(event, game) {
    if (event.decision === 'move') {
      const coords = event.move?.coords;
      const tile = event.move?.tile || coordsToTile(coords);
      const shipValue = Array.isArray(coords) ? game.true_board?.[coords[0]]?.[coords[1]] : null;
      const hit = typeof shipValue === 'number' && shipValue > 0 && event.board?.[coords[0]]?.[coords[1]] === shipValue;
      const result = hit ? `Hit ${SHIP_NAMES[shipValue] || `Ship ${shipValue}`}` : 'Miss';
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

  createEventSummary(event, game) {
    if (event.decision === 'move') {
      const coords = event.move?.coords;
      const tile = event.move?.tile || coordsToTile(coords);
      const shipValue = Array.isArray(coords) ? game.true_board?.[coords[0]]?.[coords[1]] : null;
      const hit = typeof shipValue === 'number' && shipValue > 0 && event.board?.[coords[0]]?.[coords[1]] === shipValue;
      if (hit) {
        return `Shot ${tile} → Hit ${SHIP_NAMES[shipValue] || `Ship ${shipValue}`}`;
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

  updateFrameAndEventLabels(game, event) {
    if (this.elements.frameLabel) {
      this.elements.frameLabel.textContent = `Turn ${this.state.currentStage + 1} of ${game.events.length}`;
    }
    if (this.elements.eventLabel) {
      this.elements.eventLabel.textContent = this.createEventSummary(event, game);
    }
    if (this.elements.boardCaption) {
      if (this.state.currentView === 'captain') {
        this.elements.boardCaption.textContent = `Captain view after turn ${this.state.currentStage + 1}`;
      } else {
        this.elements.boardCaption.textContent = 'Spotter view (hatched = hidden from the captain)';
      }
    }
  }

  renderMetrics(game) {
    const metricsEl = this.elements.metrics;
    if (!metricsEl) return;
    const metrics = [
      { label: 'Outcome', value: game.is_won ? 'Win' : 'Loss' },
      { label: 'F1 Score', value: formatDecimal(game.f1_score) },
      { label: 'Questions', value: `${game.question_count ?? '–'}` },
      { label: 'Moves', value: `${game.move_count ?? '–'}` },
    ];
    metricsEl.innerHTML = '';
    metrics.forEach((metric) => {
      const card = document.createElement('div');
      card.className = 'metric-card';

      const label = document.createElement('span');
      label.textContent = metric.label;

      const value = document.createElement('strong');
      value.textContent = metric.value;

      card.appendChild(label);
      card.appendChild(value);
      metricsEl.appendChild(card);
    });
  }

  buildTimeline(game) {
    const listEl = this.elements.timelineList;
    if (!listEl) return;
    this.timelineButtons = [];
    listEl.innerHTML = '';
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
      textSpan.textContent = this.createTimelineText(event, game);

      summaryWrapper.appendChild(typeSpan);
      summaryWrapper.appendChild(textSpan);

      button.appendChild(indexBadge);
      button.appendChild(summaryWrapper);

      button.addEventListener('click', () => {
        this.setStage(index, { scrollTimeline: true });
      });

      listEl.appendChild(button);
      this.timelineButtons.push(button);
    });
  }

  highlightActiveTimeline() {
    let targetButton = null;
    this.timelineButtons.forEach((button, index) => {
      const isActive = index === this.state.currentStage;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      if (isActive) {
        targetButton = button;
      }
    });
    if (this.timelineScrollRequested && targetButton) {
      const activeElement = document.activeElement;
      const shouldScroll = activeElement && activeElement !== document.body && this.root.contains(activeElement);
      if (shouldScroll) {
        requestAnimationFrame(() => {
          targetButton.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        });
      }
    }
    this.timelineScrollRequested = false;
  }

  renderStageDependent() {
    const game = this.getCurrentGame();
    const event = this.getCurrentEvent();
    if (!game || !event) {
      const fallback = Promise.resolve();
      this.lastRenderedStageIndex = this.state.currentStage;
      this.lastEventAnimationPromise = fallback;
      return fallback;
    }
    this.renderBoard(game, event);
    this.renderShipTracker(game, event.board);
    const animationPromise = this.renderEventDetails(game, event);
    this.lastRenderedStageIndex = this.state.currentStage;
    this.lastEventAnimationPromise = Promise.resolve(animationPromise);
    this.renderProgress(game, event.board);
    this.updateFrameAndEventLabels(game, event);
    this.highlightActiveTimeline();
    if (this.elements.slider) {
      this.elements.slider.setAttribute('aria-valuenow', String(this.state.currentStage));
    }
    return this.lastEventAnimationPromise;
  }

  setGame(index, options = {}) {
    const games = this.state.data?.games;
    if (!Array.isArray(games) || games.length === 0) return;
    const clampedIndex = Math.max(0, Math.min(index, games.length - 1));
    this.state.currentGameIndex = clampedIndex;
    this.state.currentStage = 0;
    const game = this.getCurrentGame();
    if (!game) return;
    this.state.hasActiveGame = true;
    this.updateStatusForGame(game);
    this.renderMetrics(game);
    this.buildTimeline(game);
    this.updateSlider(game);
    this.timelineScrollRequested = true;
    this.highlightActiveGameButton();
    this.updateGameNavButtons();

    const animationPromise = this.renderStageDependent();

    if (!options.maintainPlayback) {
      this.setPlaying(false);
    } else {
      this.clearPlaybackTimer();
      this.cancelSliderAnimation();
      if (this.playback.isPlaying) {
        this.schedulePlaybackAfterEvent(animationPromise, game, this.state.currentStage);
      }
    }
  }

  setStage(stage, options = {}) {
    const game = this.getCurrentGame();
    if (!game || !Array.isArray(game.events) || game.events.length === 0) return;
    const clampedStage = Math.max(0, Math.min(stage, game.events.length - 1));
    const prevStage = this.state.currentStage;
    const {
      scrollTimeline = true,
      animateSlider = false,
      animationDuration,
    } = options;
    const effectiveDuration = typeof animationDuration === 'number'
      ? animationDuration
      : this.playback.intervalMs;
    this.timelineScrollRequested = scrollTimeline && clampedStage !== prevStage;
    this.state.currentStage = clampedStage;
    if (this.elements.slider) {
      if (animateSlider && effectiveDuration > 0 && clampedStage !== prevStage) {
        this.enableSliderStepless();
        this.animateSliderValue(prevStage, clampedStage, effectiveDuration, () => {
          if (!this.playback.isPlaying) {
            this.disableSliderStepless();
          }
        });
      } else {
        this.cancelSliderAnimation();
        this.setSliderValue(clampedStage);
        if (!this.playback.isPlaying) {
          this.disableSliderStepless();
        }
      }
      this.elements.slider.setAttribute('aria-valuenow', String(clampedStage));
    }
    const animationPromise = this.renderStageDependent();
    if (this.playback.isPlaying) {
      this.schedulePlaybackAfterEvent(animationPromise, game, clampedStage);
    } else {
      this.clearPlaybackTimer();
    }
  }

  updateViewButtons() {
    if (this.elements.viewCaptain) {
      const isActive = this.state.currentView === 'captain';
      this.elements.viewCaptain.classList.toggle('is-active', isActive);
      this.elements.viewCaptain.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    }
    if (this.elements.viewSpotter) {
      const isActive = this.state.currentView === 'spotter';
      this.elements.viewSpotter.classList.toggle('is-active', isActive);
      this.elements.viewSpotter.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    }
  }

  setView(view) {
    if (this.state.currentView === view) return;
    this.state.currentView = view;
    this.updateViewButtons();
    this.renderStageDependent();
  }

  clearViewer() {
    this.clearPlaybackTimer();
    this.playback.isPlaying = false;
    this.updatePlayButton();
    this.state.currentStage = 0;
    this.state.hasActiveGame = false;
    this.cancelSliderAnimation();
    this.disableSliderStepless();

    if (this.elements.metrics) {
      this.elements.metrics.innerHTML = '';
    }
    if (this.elements.board) {
      this.elements.board.innerHTML = '';
    }
    if (this.elements.boardCaption) {
      this.elements.boardCaption.textContent = 'Select a game to view the board.';
    }
    if (this.elements.shipTracker) {
      this.elements.shipTracker.innerHTML = '';
    }
    if (this.elements.eventDetails) {
      this.elements.eventDetails.innerHTML = '';
    }
    if (this.elements.progressDetails) {
      this.elements.progressDetails.innerHTML = '';
    }
    if (this.elements.timelineList) {
      this.elements.timelineList.innerHTML = '';
    }
    if (this.elements.frameLabel) {
      this.elements.frameLabel.textContent = '';
    }
    if (this.elements.eventLabel) {
      this.elements.eventLabel.textContent = '';
    }
    if (this.elements.slider) {
      this.setSliderValue(0);
      this.elements.slider.disabled = true;
      this.elements.slider.max = '0';
      this.elements.slider.setAttribute('aria-valuemax', '0');
      this.elements.slider.setAttribute('aria-valuenow', '0');
    }

    this.timelineButtons = [];
    this.updateGameNavButtons();
  }

  populateFilterOptions(games) {
    if (!this.elements.filterLLM && !this.elements.filterType) {
      return;
    }
    const llmValues = sortByPreferredOrder(getUniqueValues(games, 'captain_llm'), LLM_ORDER);
    const typeValues = sortByPreferredOrder(getUniqueValues(games, 'captain_type'), CAPTAIN_TYPE_ORDER);

    const renderFilterOptions = (selectEl, values, defaultLabel, labels = {}) => {
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
    };

    renderFilterOptions(this.elements.filterLLM, llmValues, 'All LLMs', LLM_LABELS);
    renderFilterOptions(this.elements.filterType, typeValues, 'All strategies', CAPTAIN_TYPE_LABELS);

    if (this.elements.filterLLM) {
      const desired = this.state.filters.llm;
      const value = desired && llmValues.includes(desired) ? desired : '';
      this.elements.filterLLM.value = value;
      this.state.filters.llm = this.elements.filterLLM.value || '';
    }

    if (this.elements.filterType) {
      const desired = this.state.filters.type;
      const value = desired && typeValues.includes(desired) ? desired : '';
      this.elements.filterType.value = value;
      this.state.filters.type = this.elements.filterType.value || '';
    }
  }

  renderGameList() {
    if (!this.elements.gameList) return;
    this.elements.gameList.innerHTML = '';
    this.gameButtons = new Map();

    if (!this.state.filteredIndices.length) {
      const message = document.createElement('div');
      message.className = 'empty-state';
      message.textContent = 'No games match these filters.';
      this.elements.gameList.appendChild(message);
      this.elements.gameList.removeAttribute('aria-activedescendant');
      this.updateGameNavButtons();
      return;
    }

    this.state.filteredIndices.forEach((gameIndex) => {
      const game = this.state.data?.games?.[gameIndex];
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
        this.setGame(gameIndex);
      });

      this.elements.gameList.appendChild(button);
      this.gameButtons.set(gameIndex, button);
    });

    this.highlightActiveGameButton();
    this.updateGameNavButtons();
  }

  highlightActiveGameButton() {
    if (!this.elements.gameList) return;
    let activeId = null;
    this.gameButtons.forEach((button, index) => {
      const isActive = index === this.state.currentGameIndex;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
      if (isActive) {
        activeId = button.id;
      }
    });
    if (activeId) {
      this.elements.gameList.setAttribute('aria-activedescendant', activeId);
    } else {
      this.elements.gameList.removeAttribute('aria-activedescendant');
    }
  }

  applyFilters(options = {}) {
    const { preserveSelection = true } = options;
    const games = this.state.data?.games ?? [];
    const filtered = [];

    games.forEach((game, index) => {
      if (!game) return;
      const matchesLLM = !this.state.filters.llm || game.captain_llm === this.state.filters.llm;
      const matchesType = !this.state.filters.type || game.captain_type === this.state.filters.type;
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

    this.state.filteredIndices = filtered;
    this.renderGameList();

    if (!filtered.length) {
      this.setStatusMessage('No games match the selected filters. Try a different combination.', true);
      this.clearViewer();
      return;
    }

    const preserveCurrent = preserveSelection && this.state.hasActiveGame && filtered.includes(this.state.currentGameIndex);
    if (preserveCurrent) {
      const game = this.getCurrentGame();
      if (game) {
        this.updateStatusForGame(game);
        this.highlightActiveGameButton();
        this.updateGameNavButtons();
      }
    } else {
      this.setGame(filtered[0]);
    }
  }

  attachEventListeners() {
    if (this.elements.filterLLM) {
      this.elements.filterLLM.addEventListener('change', (event) => {
        this.state.filters.llm = event.target.value || '';
        this.applyFilters();
      });
    }

    if (this.elements.filterType) {
      this.elements.filterType.addEventListener('change', (event) => {
        this.state.filters.type = event.target.value || '';
        this.applyFilters();
      });
    }

    if (this.elements.slider) {
      this.elements.slider.addEventListener('input', (event) => {
        const rawValue = Number(event.target.value);
        const stage = Number.isNaN(rawValue) ? 0 : Math.round(rawValue);
        if (!Number.isNaN(rawValue) && rawValue !== stage) {
          event.target.value = String(stage);
        }
        this.setStage(stage, { scrollTimeline: false });
      });
    }

    if (this.elements.viewCaptain) {
      this.elements.viewCaptain.addEventListener('click', () => this.setView('captain'));
    }

    if (this.elements.viewSpotter) {
      this.elements.viewSpotter.addEventListener('click', () => this.setView('spotter'));
    }

    if (this.elements.playButton) {
      this.elements.playButton.addEventListener('click', () => {
        this.setPlaying(!this.playback.isPlaying);
      });
    }

    if (this.elements.timelineReset) {
      this.elements.timelineReset.addEventListener('click', () => {
        const game = this.getCurrentGame();
        if (!game || !Array.isArray(game.events) || !game.events.length) return;
        this.setStage(0);
      });
    }

    if (this.elements.timelineSkip) {
      this.elements.timelineSkip.addEventListener('click', () => {
        const game = this.getCurrentGame();
        if (!game || !Array.isArray(game.events) || !game.events.length) return;
        this.setStage(game.events.length - 1);
        this.setPlaying(false);
      });
    }

    if (this.elements.prevGame) {
      this.elements.prevGame.addEventListener('click', () => {
        this.changeGame(-1);
      });
    }

    if (this.elements.nextGame) {
      this.elements.nextGame.addEventListener('click', () => {
        this.changeGame(1);
      });
    }

    if (this.elements.speedSelect) {
      this.elements.speedSelect.addEventListener('change', (event) => {
        const value = Number(event.target.value);
        if (!Number.isFinite(value) || value <= 0) return;
        this.setPlaybackSpeed(value);
      });
    }
  }

  getNextFilteredIndex(offset = 1) {
    if (!this.state.filteredIndices.length) return null;
    const currentPosition = this.state.filteredIndices.indexOf(this.state.currentGameIndex);
    const position = currentPosition === -1 ? 0 : currentPosition;
    let nextPosition = position + offset;
    if (this.config.loopGames) {
      nextPosition = (nextPosition + this.state.filteredIndices.length) % this.state.filteredIndices.length;
    }
    if (nextPosition < 0 || nextPosition >= this.state.filteredIndices.length) {
      return null;
    }
    return this.state.filteredIndices[nextPosition];
  }

  advanceToNextGame() {
    const nextIndex = this.getNextFilteredIndex(1);
    if (nextIndex === null) {
      return false;
    }
    const wasPlaying = this.playback.isPlaying;
    this.setGame(nextIndex, { maintainPlayback: true });
    if (wasPlaying) {
      this.playback.isPlaying = true;
      this.updatePlayButton();
      const game = this.getCurrentGame();
      const animationPromise = this.renderStageDependent();
      this.schedulePlaybackAfterEvent(animationPromise, game, this.state.currentStage);
    }
    return true;
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
      text: "That's impressive! Can you give an example?",
      delayAfter: 320,
      iconClass: 'fas fa-book-reader',
    },
    {
      id: 'authors-example',
      role: 'spotter',
      label: 'Authors',
      text: "Sure! For example, we find that Llama-4-Scout is barely above random chance. However, with our methods, Llama-4-Scout improves dramatically at both asking and answering questions -- we see win rates of 82% against humans and 67% against GPT-5. This is especially exciting given that Llama-4-Scout costs about 100x less to run than GPT-5; our findings suggest that there's a lot of room to build more efficient AI systems.",
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
      text: "That's great to hear! How can I learn more about this work?",
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

function initHeroExplorer(dataPromise) {
  const heroRoot = document.getElementById('hero-trajectory');
  if (!heroRoot) return;
  new TrajectoryExplorer(heroRoot, {
    selectors: {
      board: '[data-role="hero-board"]',
      eventDetails: '[data-role="hero-event-details"]',
      status: null,
      filterLLM: null,
      filterType: null,
      gameList: null,
      viewCaptain: null,
      viewSpotter: null,
      metrics: null,
      boardCaption: '[data-role="hero-board-caption"]',
      slider: null,
      frameLabel: null,
      eventLabel: null,
      progressDetails: null,
      shipTracker: null,
      timelineList: null,
      playButton: null,
      timelineReset: null,
      timelineSkip: null,
      prevGame: null,
      nextGame: null,
      speedSelect: null,
    },
    initialView: 'captain',
    autoplay: true,
    playbackSpeed: 1,
    autoAdvanceGames: true,
    loopGames: true,
    dataPromise,
  });
}

function initMainExplorer(dataPromise) {
  const explorerRoot = document.getElementById('trajectory-explorer');
  if (!explorerRoot) return;
  new TrajectoryExplorer(explorerRoot, { dataPromise });
}

document.addEventListener('DOMContentLoaded', () => {
  const dataPromise = getTrajectoryData();
  initHeroExplorer(dataPromise);
  initMainExplorer(dataPromise);
  initMotivationChat();
});
