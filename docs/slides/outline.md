# Talk Outline — "Shoot First, Ask Questions Later?"

ICLR 2026 | 10 min talk + 2 min Q&A

---

## 1. Title + Motivation (1.5 min)

- **Title slide**
- **The promise**: LM agents are being deployed for scientific discovery, medical diagnosis, agentic coding — domains that require forming hypotheses, gathering information, and acting under uncertainty
  - Nod to high-profile examples (e.g., FutureHouse) — exciting but hard to verify and harder to reproduce in a benchmark
- **The evaluation gap**: True discovery is hard to benchmark because LMs already have strong priors about real-world domains. We need controlled settings where we can formalize what "rational" means
- **Core tension**: LMs are trained to *answer* questions, but discovery requires *asking* them. Can they do both?

## 2. The Collaborative Battleship Task (2 min)

- Credit Rothe, Lake & Gureckis — a well-studied paradigm from cognitive science designed to evaluate information-seeking behavior in humans. We adapt and extend it into a two-player, multi-turn dialogue task
- **Use game-explorer** to introduce the mechanics behaviorally:
  - Show the board. Captain sees partial information; Spotter sees everything
  - Each turn: Captain chooses to ask a yes/no question or shoot at a tile
  - Step through a few turns — question, answer, shot — to make roles and the explore/exploit tension concrete
  - Emphasize the information bottleneck: Spotter can only say Yes/No
- **ARC parallel**: Like ARC, this is a minimal grid-based task that proves deceptively challenging for AI — simple rules, but solving it requires genuine reasoning
- Key extensions over prior work on a slide (full multi-turn games, two-player dialogue, Python programs, 8×8 boards, yes/no bottleneck) — visible but not belabored verbally

## 3. Formal Framework: EIG (1 min)

- Show the EIG equation: "How much does the answer to this question tell us about the hidden board?"
- **Use eig-demo**: sample-sorting visualization — samples split into True/False buckets; the more balanced the split, the higher the EIG
- Show the closed-form (Eq. 4, binary entropy) on screen. Quick intuition, don't dwell

## 4. Human Behavioral Study (0.5–1 min)

- N=42 participants, 126 game trajectories → BattleshipQA dataset
- Asking questions correlates with performance (Fig. 2a)
- People show diverse explore/exploit strategies — they're resource-rational, not Bayes-optimal
- Sets up *why* we need both evaluation and methods

## 5. SpotterQA: Can LMs Answer Grounded Questions? (1.5 min)

- **Setup**: Answering questions accurately is crucial — humans achieve 92.5% accuracy, establishing a reliable baseline. How do LMs compare?
- **Hypothesis test framing**: language-based reasoning (direct answering ± CoT) vs. code-based reasoning (code generation ± CoT)
- **Use code-demo** to show one example: natural language question → Python program → executed on board → grounded answer
- **Results** (Fig. 3):
  - Wide accuracy range across 15 LMs (Fig. 3a)
  - Code generation consistently wins: +14.7% over base, closing the gap to human performance (Fig. 3b)
  - But: LMs degrade on complex/context-dependent questions where humans stay consistent (Fig. 3c) — code helps but doesn't fully solve pragmatic grounding

## 6. CaptainQA + Bayesian Strategies (2.5 min)

- **Three strategies** — show each card with formal math visible, explain the intuition:
  - **Q_Bayes**: Sample candidate questions from the LM, pick the one with highest EIG
  - **M_Bayes**: Shoot at the tile most likely to contain a ship (MAP over belief)
  - **D_Bayes**: One-step lookahead — compare "ask then shoot" vs. "shoot now"
- **Redundant question example**: Show a concrete EIG=0 question (e.g., asking something already determined by prior answers). Motivates why Q_Bayes matters — weaker models ask 15–18% redundant questions; Q_Bayes eliminates these
- **Key results** (Fig. 4):
  - Fig. 4a: Bayesian strategies lift even Llama-4-Scout from near-random to super-human F1 (0.367 → 0.764); win rates of 81–82% against humans, 67% against GPT-5
  - Fig. 4b: EIG scales with number of candidate samples — inference-time scaling for question quality
  - Fig. 4c: D_Bayes makes question-timing match human/GPT-5 patterns — skilled players ask *some* questions first but not all
  - Cost: Llama-4-Scout + Bayes beats GPT-5 at ≈1% of the cost

## 7. Generalization: Guess Who? (0.5 min)

- One slide: same strategies, different domain (richer object-relational semantics), similar gains
- Not Battleship-specific — generalizes to information-seeking games with combinatorial hypothesis spaces

## 8. Takeaways + Close (1 min)

- **Necessary but not sufficient**: Good performance on controlled tasks like Battleship doesn't guarantee real-world discovery — but it's a prerequisite. If an agent can't explore rationally here, we shouldn't trust it in the clinic or the lab
- **Formalizing rationality matters**: In domains like science and medicine, we need to be able to define and measure what "rational" exploration looks like. Environments like this let us do that, and compare LM behavior against human and ideal baselines
- **The world model as an interactive tool**: The symbolic world model in this work is essentially a simulator — the agent can test hypotheses by running programs against sampled possible worlds. This connects to a broader trend:
  - Coding agents using sandboxed environments to simulate consequences of changes
  - Dynamic world models (e.g., Genie) enabling robots to simulate actions before executing
  - The key idea: *give agents the ability to think before they act*, via simulation
- Point to interactive website + code

---

## Interactive Components Plan

| Section | Component | Purpose |
|---------|-----------|---------|
| 2. Battleship Task | `game-explorer.html` iframe | Introduce task mechanics behaviorally |
| 3. EIG Framework | `eig-demo.html` iframe | Visualize sample sorting + EIG intuition |
| 5. SpotterQA | `code-demo.html` iframe | Show code generation pipeline |

## Key Figures

| Figure | Section | Content |
|--------|---------|---------|
| Fig. 2a | 4. Human Study | Questions asked vs. F1 correlation |
| Fig. 3a,b,c | 5. SpotterQA | Accuracy by LM, by strategy, by complexity |
| Fig. 4a,b,c | 6. CaptainQA | F1 by strategy, EIG scaling, question timing |
| Fig. 5 | 7. Guess Who? | Success rate with Bayesian strategies |
