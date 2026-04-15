# Slides — "Shoot First, Ask Questions Later?"

ICLR 2026 presentation slides built with [reveal.js](https://revealjs.com/) v5.1.0.

## Structure

```
docs/slides/
├── index.html           # Main slide deck (reveal.js)
├── game-explorer.html   # Iframe: interactive trajectory replay
├── code-demo.html       # Iframe: code generation showcase
├── eig-demo.html        # Iframe: EIG sample-sorting visualization
└── README.md
```

The three `*-demo.html` / `game-explorer.html` files are standalone pages that
embed interactive components from the parent website (`docs/`). They are loaded
into the deck via `<iframe data-src="...">` with lazy loading.

Each frame page uses `<base href="../">` so that the website's JS
(`static/js/index.js`) resolves its relative asset paths (e.g.
`./static/data/trajectory_samples.json`) against the `docs/` root.

## Local development

Serve from the **`battleship/`** repo root (one level above `docs/`):

```bash
cd /path/to/battleship
python3 -m http.server 8765
```

Then open: <http://localhost:8765/docs/slides/>

> You need to serve from the repo root (or at least from `docs/`) because the
> iframe pages fetch data and assets via relative paths from `docs/`.

## Presenter controls (reveal.js)

| Key | Action |
|-----|--------|
| `S` | Open speaker-notes window |
| `F` | Fullscreen |
| `O` / `Esc` | Slide overview |
| `B` / `.` | Black-out screen |
| Arrow keys | Navigate |

Slide numbers are shown as `current / total` in the bottom-right corner.

## Dependencies (all CDN)

- reveal.js 5.1.0
- Google Sans + Noto Sans fonts
- Font Awesome 6.7.2
- (via iframes) GSAP 3.12.5, Bulma 0.9.4, jQuery 3.6.4, highlight.js 11.9.0

## Branch workflow

Slides are developed on the **`slides`** branch. When the talk is ready, merge
into `main` and add a link from the website landing page.
