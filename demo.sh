#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# h5i full-workflow demo — DNN from scratch with Claude
#
# This script re-creates the entire development session from a bare directory:
#   git init → h5i init → iterate (write code → test → h5i commit with AI meta)
#
# Prerequisites (one-time):
#   pip install pytest numpy
#   cargo install --git https://github.com/<you>/h5i   # or: cargo install --path ..
#
# Usage:
#   bash demo.sh            # build full history from scratch in ./demo-run/
#   bash demo.sh --inspect  # just print h5i log / blame on the CURRENT repo
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

H5I_ADAPTER="$(dirname "$0")/../h5i/script/h5i-pytest-adapter.py"
# Fall back to sibling repo layout used as a submodule
[ -f "$H5I_ADAPTER" ] || H5I_ADAPTER="$(dirname "$0")/../../script/h5i-pytest-adapter.py"

INSPECT_ONLY=false
[[ "${1:-}" == "--inspect" ]] && INSPECT_ONLY=true

# ── helpers ──────────────────────────────────────────────────────────────────

hr() { printf '\n%s\n\n' "$(printf '─%.0s' {1..72})"; }

run_tests_and_commit() {
  local files="$1"   # space-separated paths to git add
  local message="$2"
  local prompt="$3"
  local test_filter="${4:-}"   # optional pytest path filter

  # Stage
  # shellcheck disable=SC2086
  git add $files

  # Capture test metrics
  local results
  results="$(mktemp /tmp/h5i-results-XXXXXX.json)"
  python3 "$H5I_ADAPTER" ${test_filter:+$test_filter} 2>/dev/null > "$results" || true

  echo "  pytest: $(python3 -c "import json; d=json.load(open('$results')); print(d['summary'])")"

  h5i commit \
    --message "$message" \
    --agent   "claude-code" \
    --model   "claude-sonnet-4-6" \
    --prompt  "$prompt" \
    --test-results "$results"

  rm -f "$results"
}

# ── inspect-only mode ────────────────────────────────────────────────────────

if $INSPECT_ONLY; then
  hr
  echo "h5i log (full history)"
  hr
  h5i log --limit 20

  hr
  echo "h5i blame dnn/network.py"
  hr
  h5i blame dnn/network.py

  hr
  echo "python examples/train_xor.py"
  hr
  python3 examples/train_xor.py
  exit 0
fi

# ── build fresh history in a temp directory ───────────────────────────────────

WORKDIR="$(mktemp -d /tmp/h5i-dnn-demo-XXXXXX)"
echo "Building demo in: $WORKDIR"

cp -r "$(dirname "$0")/." "$WORKDIR/"
cd "$WORKDIR"

# Remove any existing git history so we start clean
rm -rf .git

git init -q
git config user.name  "Claude Sonnet 4.6"
git config user.email "claude@anthropic.com"
h5i init

hr; echo "Step 0 — project scaffold"
run_tests_and_commit \
  ".gitignore pyproject.toml" \
  "chore: initialise dnn-from-scratch project" \
  "Set up a Python project for a fully-connected DNN with SGD from scratch using NumPy"

hr; echo "Step 1 — Linear layer"
run_tests_and_commit \
  "dnn/__init__.py dnn/layers.py tests/__init__.py tests/test_layers.py" \
  "feat: implement Linear layer with He initialisation" \
  "Implement a fully-connected Linear layer class with He initialisation, forward pass (y = xW + b), backward pass that computes dW, db and grad w.r.t. input, and a parameters() method for the optimiser" \
  "tests/test_layers.py"

hr; echo "Step 2 — Activations"
run_tests_and_commit \
  "dnn/activations.py tests/test_activations.py" \
  "feat: add ReLU and Sigmoid activations" \
  "Add element-wise activation functions: ReLU (max(0,x)) and Sigmoid (1/(1+exp(-x))) with their analytic backward passes" \
  "tests/test_activations.py"

hr; echo "Step 3 — Loss functions"
run_tests_and_commit \
  "dnn/losses.py tests/test_losses.py" \
  "feat: add MSE and BCE loss functions" \
  "Implement MSELoss and BCELoss with numerically stable forward/backward passes and finite-difference gradient verification" \
  "tests/test_losses.py"

hr; echo "Step 4 — SGD optimiser"
run_tests_and_commit \
  "dnn/optimizers.py tests/test_optimizers.py" \
  "feat: implement SGD optimiser with optional momentum" \
  "Implement SGD optimiser with optional momentum coefficient. Should raise ValueError for invalid lr or momentum, skip layers with None gradients, and accumulate velocity across steps when momentum > 0" \
  "tests/test_optimizers.py"

hr; echo "Step 5 — Sequential network + XOR example"
run_tests_and_commit \
  "dnn/network.py tests/test_network.py examples/train_xor.py" \
  "feat: add Sequential container and XOR training example" \
  "Implement a Sequential model container that chains layers, add an end-to-end XOR convergence test, and provide a training script showing the full forward/backward/step loop"

# ── final inspection ──────────────────────────────────────────────────────────

hr
echo "h5i log — full 5-D provenance history"
hr
h5i log --limit 10

hr
echo "h5i blame — dnn/network.py"
hr
h5i blame dnn/network.py

hr
echo "python examples/train_xor.py"
hr
python3 examples/train_xor.py

hr
echo "Demo complete — repo at: $WORKDIR"
echo "  h5i log, h5i blame, h5i diff, h5i rollback all work there."
