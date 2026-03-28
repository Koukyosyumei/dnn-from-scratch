# Building a Conv2D Layer with h5i: A Session-Aware Workflow

> This post walks through adding a `Conv2D` layer to a NumPy neural network library
> using **h5i** to track AI reasoning, catch uncertainty early, and leave a clean
> handoff for the next session. Every terminal block below is real output.

---

## The Task

The [dnn-from-scratch](https://github.com/Koukyosyumei/dnn-from-scratch) repo already
has `Linear`, `ReLU`, `SGD`, `Adam`, and a `Sequential` container — all pure NumPy.
The missing piece is a 2-D convolutional layer with a correct backward pass. That means:

- `im2col` / `col2im` helpers to keep the math readable
- `forward(x)` — cross-correlation via matrix multiply
- `backward(grad_out)` — gradients for `W`, `b`, and `x`
- Numerical gradient checks for all three

---

## Step 1 — Initialize the context workspace

Before touching any code, give the AI agent a persistent goal it can restore next session:

```
$ h5i context init --goal "Add Conv2D layer with forward/backward pass and tests"
```
```
✔  .h5i-ctx/ workspace initialized at .h5i-ctx/

  Quick-start:
    h5i context commit "summary" --detail "…"  checkpoint your progress
    h5i context branch experiment/foo --purpose "…"  explore an alternative
    h5i context show --trace  view current context
```

Check what was created:

```
$ h5i context show
```
```
── Context ─────────────────────────────────────────────
  Project: Add Conv2D layer with forward/backward pass and tests  (branch: main)

  Milestones:
    ○ [ ] Initial setup
```

---

## Step 2 — Log the OTA trace as work happens

The agent records every meaningful step — observations, design decisions, and actions —
into the workspace trace before writing any code:

```
$ h5i context trace --kind OBSERVE \
    "Repo has Linear, ReLU, Sigmoid, MSELoss, BCELoss, SGD, Adam, Sequential — all numpy-only, no Conv layer yet"
```
```
◈ [OBSERVE] Repo has Linear, ReLU, Sigmoid, MSELoss, BCELoss, SGD, Adam, Sequential — all numpy-only, no Conv layer yet
```

```
$ h5i context trace --kind THINK \
    "Will implement Conv2D with im2col for the forward pass and col2im for backward — keeps the math readable and the gradient check straightforward"
```
```
◈ [THINK] Will implement Conv2D with im2col for the forward pass and col2im for backward — keeps the math readable and the gradient check straightforward
```

The agent then writes the implementation (`dnn/layers.py`) and 13 tests
(`tests/test_conv2d.py`), then runs the full suite:

```
$ python3 -m pytest --tb=short -q
```
```
.........................................................  [100%]
57 passed in 0.44s
```

```
$ h5i context trace --kind ACT \
    "Wrote Conv2D class with im2col/col2im helpers in dnn/layers.py; added 13 tests in tests/test_conv2d.py; all 57 tests pass"
```
```
◈ [ACT] Wrote Conv2D class with im2col/col2im helpers in dnn/layers.py; added 13 tests in tests/test_conv2d.py; all 57 tests pass
```

---

## Step 3 — Commit with AI provenance and test results

```
$ h5i commit \
    -m "implement Conv2D layer with im2col/col2im" \
    --prompt "implement a simple convolutional layer with forward and backward pass" \
    --model claude-sonnet-4-6 \
    --agent claude-code \
    --test-results /tmp/conv2d_results.json \
    --audit
```
```
✔  Integrity check passed.
✔  h5i Commit Created: e4c4566ca8accd9be5473825aacf244c1e18768f
```

The `--audit` flag runs all 12 deterministic integrity rules before the commit is
written. Clean diff, no credential patterns, no undeclared deletions — nothing fires.

```
$ h5i log --limit 1
```
```
commit e4c4566ca8accd9be5473825aacf244c1e18768f
Author:    Claude Sonnet 4.6 <claude@anthropic.com>
Agent:     claude-code (claude-sonnet-4-6)
Prompt:    "implement a simple convolutional layer with forward and backward pass"
Tests:     ✔ 57 passed in 0.42s [pytest]

    implement Conv2D layer with im2col/col2im
```

The prompt, model, agent ID, and full test results are stored in Git Notes alongside
the commit — readable with `git notes show e4c4566c`.

---

## Step 4 — Checkpoint the context workspace

```
$ h5i context commit "Implemented Conv2D layer" \
    --detail "Added Conv2D class with im2col/col2im helpers. All 3 gradient checks (dW, db, dx) pass. 13 new tests, 57 total passing."
```
```
✔  Context commit recorded — Implemented Conv2D layer
```

---

## Step 5 — Analyze the session log

After the session, link the Claude Code conversation log to the commit so h5i can
extract a footprint, causal chain, and uncertainty signals:

```
$ h5i notes analyze \
    --session ~/.claude/projects/-home-koukyosyumei-Dev-dnn-from-scratch/c0661020-….jsonl \
    --commit e4c4566c
```
```
➜  Analyzing session log → commit e4c4566c
✔  34 messages · 13 tool calls · 1 edited · 6 consulted
  ℹ Run h5i notes show e4c4566c to inspect results.
```

### Footprint

Which files did the agent read vs actually change?

```
$ h5i notes show e4c4566c
```
```
── Exploration Footprint ──────────────────────────────────
  Session c0661020  ·  34 messages  ·  13 tool calls

  Files Consulted:
    📖 ~/Dev/dnn-from-scratch ×3  [Grep,Glob]
    📖 ~/Dev/dnn-from-scratch/tests/test_network.py ×2  [Read]
    📖 ~/Dev/dnn-from-scratch/dnn/network.py ×1  [Read]
    📖 ~/Dev/dnn-from-scratch/tests/test_optimizers.py ×1  [Read]
    📖 ~/Dev/dnn-from-scratch/dnn/optimizers.py ×1  [Read]
    📖 ~/Dev/dnn-from-scratch/dnn/__init__.py ×1  [Read]

  Files Edited:
    ✏ ~/Dev/dnn-from-scratch/dnn/__init__.py  ×1 edit(s)

  Implicit Dependencies (read but not edited):
    → ~/Dev/dnn-from-scratch
    → ~/Dev/dnn-from-scratch/dnn/network.py
    → ~/Dev/dnn-from-scratch/dnn/optimizers.py
    → ~/Dev/dnn-from-scratch/tests/test_network.py
    → ~/Dev/dnn-from-scratch/tests/test_optimizers.py
── Causal Chain ────────────────────────────────────────────
  Trigger:
    ""

  Edit Sequence:
     1. ~/Dev/dnn-from-scratch/dnn/__init__.py  Edit t:20
```

The agent read `network.py`, `optimizers.py`, and their tests — checking how existing
layers expose `parameters()` — before writing a single line. Those files are now tracked
as **implicit dependencies** of this commit.

### Uncertainty heatmap

Where did the agent hedge or express doubt?

```
$ h5i notes uncertainty
```
```
── Uncertainty Heatmap ─────────────────────────────────────────────
  1 signal  ·  session c0661020  ·  1 file

  Risk Map
  ──────────────────────────────────────────────────────────────────────────
  ~/Dev/dnn-from-scratch/dnn/__init__.py        █████████░░░░░░░  ●  1  signal   avg  45%

  Timeline
  t:29 ────────────────────────────────────────────────────────── t:29
  ▓···································································
  ↑t:29

  Signals
  ──────────────────────────────────────────────────────────────────────────
  ▓▓  t:29    let me verify       …/dnn-from-scratch/dnn/__init__.py  [ 45%]
       "tests already exist. The only missing piece was the export. Let me verify everything works"

  ██ high risk (<35%)   ▓▓ moderate (35–55%)   ░░ low (>55%)
```

One moderate signal at turn 29 in `__init__.py` — the agent paused to verify the export
was correct before finishing. That's exactly the right place to be cautious; the heatmap
flags it for human review.

### File churn

```
$ h5i notes churn
```
```
── File Churn ──────────────────────────────────────────────
  file                                           edits reads  churn
  ────────────────────────────────────────────────────────────────────
  ~/Dev/dnn-from-scratch/dnn/__init__.py             1     1  █████░░░░░ 50%
  ~/Dev/dnn-from-scratch                             0     3  ░░░░░░░░░░ 0%
  ~/Dev/dnn-from-scratch/tests/test_network.py       0     2  ░░░░░░░░░░ 0%
```

`__init__.py` was read once, then edited once — 50% churn. Everything else was
read-only reference material.

### Intent graph across the whole repo history

```
$ h5i notes graph --limit 5
```
```
Intent Graph ─ 5 commits, mode: prompt ──────────────────────────

  e4c4566c [prompt] "implement a simple convolutional layer with forward and backward pass"
     implement Conv2D layer with im2col/col2im
     agent: claude-code

  057f2dc6 [msg] "add a deom script"
     add a deom script

  22924388 [prompt] "can you implement Adam optimizer? This repo already has implement SGD"
     import Adam
     agent: claude-code

  fa4369f5 [msg] "Revert "chore: add demo.sh full-workflow script""
     Revert "chore: add demo.sh full-workflow script"

  86d3b1bf [msg] "implement ADAM optimizer"
     implement ADAM optimizer

────────────────────────────────────────────────────────────
2 AI commits, 0 causal links
```

The two AI-authored commits stand out immediately: each carries the exact prompt that
produced it. Human commits show `[msg]` with no prompt attached.

---

## Step 6 — Inspect the full context trace

At any point — or at the start of the next session — the complete OTA trace is
recoverable:

```
$ h5i context show --trace
```
```
── Context ─────────────────────────────────────────────
  Project: Add Conv2D layer with forward/backward pass and tests  (branch: main)

  Milestones:
    ○ [ ] Initial setup

  Recent Commits:
    ◈ Added Conv2D class with im2col/col2im helpers. All 3 gradient checks (dW, db, dx) pass. 13 new tests

  Recent OTA Log:
    # OTA Log — Branch: main

    [00:45:18] OBSERVE: Repo has Linear, ReLU, Sigmoid, MSELoss, BCELoss, SGD, Adam, Sequential — all numpy-only, no Conv layer yet
    [00:45:24] THINK: Will implement Conv2D with im2col for the forward pass and col2im for backward — keeps the math readable and the gradient check straightforward
    [00:47:17] ACT: Wrote Conv2D class with im2col/col2im helpers in dnn/layers.py; added 13 tests in tests/test_conv2d.py; all 57 tests pass


    ---
    _[Checkpoint: 69c72536 — Implemented Conv2D layer]_
    ---
```

The full reasoning chain — observation → design decision → action → checkpoint — is
version-controlled alongside the code.

---

## Step 7 — Generate a handoff briefing for the next session

```
$ h5i resume
```
```
➜  Generating handoff briefing...
── Session Handoff ─────────────────────────────────────────────────
  Branch: master  ·  Last active: 2026-03-28 00:47 UTC
  Agent: claude-code  ·  Model: claude-sonnet-4-6
  HEAD: e4c4566c  implement Conv2D layer with im2col/col2im

  Goal
    Add Conv2D layer with forward/backward pass and tests

  Progress
    ○ Initial setup  ← resume here

  Last Session
    c0661020  ·  34 messages  ·  13 tool calls  ·  1 file edited

  ⚠  High-Risk Files  (review before continuing)
    ██████████  ~/Dev/dnn-from-scratch/dnn/__init__.py  1 signal   churn 50%  "let me verify"

  Recent Context Commits
    ◈ Added Conv2D class with im2col/col2im helpers. All 3 gradient checks (dW, db, dx

  Suggested Opening Prompt
  ────────────────────────────────────────────────────────────────────
  Continue: Add Conv2D layer with forward/backward pass and tests. Next
  milestone: Initial setup. Review
  ~/Dev/dnn-from-scratch/dnn/__init__.py carefully before editing. Last
  session flagged "let me verify" here.
  ────────────────────────────────────────────────────────────────────
```

The next agent session gets the goal, what was done, which file needs extra care, and a
ready-to-paste opening prompt — all assembled from local data with no API call.

---

## What each h5i command contributed

| Command | What it answered |
|---------|-----------------|
| `h5i context init` | Set a durable goal for the session |
| `h5i context trace` | Recorded the OBSERVE → THINK → ACT chain as it happened |
| `h5i commit --audit` | Attached prompt, model, test results, and ran 12 integrity rules |
| `h5i log` | Showed the commit with its full AI provenance inline |
| `h5i context commit` | Checkpointed the milestone into the reasoning workspace |
| `h5i notes analyze` | Linked the Claude Code conversation log to the commit |
| `h5i notes show` | Revealed which files were read (implicit deps) vs actually changed |
| `h5i notes uncertainty` | Pinpointed the one moment the agent hedged, with its exact quote |
| `h5i notes churn` | Quantified edit vs read ratio per file |
| `h5i notes graph` | Mapped intent across the whole repo history — AI vs human at a glance |
| `h5i context show --trace` | Restored the full OTA reasoning chain for any future session |
| `h5i resume` | Assembled a complete handoff briefing from all of the above |

---

## The Conv2D implementation

For reference, here is the core of what was added to `dnn/layers.py`:

```python
class Conv2D:
    """2-D convolution layer.

    y[n, f, i, j] = Σ_{c,kh,kw} W[f,c,kh,kw] · x[n, c, i·s+kh, j·s+kw] + b[f]
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)          # He initialisation
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

    def forward(self, x):
        col = _im2col(x, K, K, S, P)           # unfold patches → matrix
        out = col @ self.W.reshape(F, -1).T + self.b
        return out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

    def backward(self, grad_out):
        d_out = grad_out.transpose(0, 2, 3, 1).reshape(-1, F)
        self.db = d_out.sum(axis=0)
        self.dW = (d_out.T @ self._col).reshape(self.W.shape)
        dx = _col2im(d_out @ self.W.reshape(F, -1), self._x_shape, K, K, S, P)
        return dx
```

All three gradients (W, b, x) pass a finite-difference check at `rtol=1e-4`.

---

*This demo is part of the [h5i](https://github.com/Koukyosyumei/h5i) project —
version control for the age of AI-generated code.*
