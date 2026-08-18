#!/usr/bin/env bash
#
# Rewrite git history to remove personal and machine-specific details.
#
# Why this exists: during development, findings about one particular machine
# (a memory bank name, a port conflict caused by another local service, an
# exact hardware and OS inventory) were written into code, docs and commit
# messages. Fixing the current tree is not enough -- the old values remain
# reachable at earlier commits, which matters once a repo is public.
#
# This script is destructive and rewrites every commit SHA. It takes a full
# backup bundle first, and refuses to run against a dirty tree.
#
#   ./scripts/scrub_history.sh            # rewrite locally, then verify
#   ./scripts/scrub_history.sh --push     # ...and force-push to origin
#
# After a rewrite, anyone else with a clone must re-clone; their old history
# no longer shares ancestry with origin.
#
# NOTE ON THE ODD-LOOKING STRINGS BELOW. The search terms are assembled from
# concatenated fragments ("apo" "orva") so that this file never contains the
# literal text it is searching for. Without that, the rewrite edits its own
# source -- the terms get replaced inside this script, and the verification
# step then checks for the wrong strings. That is not hypothetical; it
# happened on the first run.

set -euo pipefail

PUSH=false
[[ "${1:-}" == "--push" ]] && PUSH=true

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# --- preconditions ---------------------------------------------------
command -v git-filter-repo >/dev/null || {
  echo "git-filter-repo is required: brew install git-filter-repo" >&2
  exit 1
}
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is dirty. Commit or stash first." >&2
  exit 1
fi

ORIGIN="$(git remote get-url origin 2>/dev/null || true)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# --- what to remove --------------------------------------------------
# Split literals; see the note in the header.
T_BANK="apo""orva"
T_RAM="128 GB"" unified"
T_OS="macOS 26.1 (Darwin ""25.1.0)"
# The local service that was named in the old text, spelled via fragments.
J="Jup""yter"
T_JUP1="$J holds"             # historic phrasings naming a local service
T_JUP2="$J usually"
T_JUP3="$J commonly"

# --- backup ----------------------------------------------------------
# A bundle is a single file containing every ref and object: enough to
# restore the pre-rewrite repo completely. Kept outside the working tree so
# the rewrite cannot touch it.
BACKUP="${TMPDIR:-/tmp}/$(basename "$REPO_ROOT")-prescrub-$(date +%Y%m%d-%H%M%S).bundle"
git bundle create "$BACKUP" --all >/dev/null 2>&1
echo "Backup written to $BACKUP"
echo "  restore with: git clone $BACKUP restored-repo"

# --- replacements ----------------------------------------------------
# Applied to file contents and commit messages at every commit. Left side is
# a literal unless prefixed with regex:.
REPLACEMENTS="$(mktemp)"
trap 'rm -f "$REPLACEMENTS"' EXIT
cat > "$REPLACEMENTS" <<REPL
${T_BANK}==>default
${T_RAM}==>unified memory
${T_OS}==>macOS on Apple Silicon
regex:${J} (commonly |usually )?(holds|owns)( that port)?==>another service may hold
regex:${J} defaults to it==>it is a common default
REPL

# --- rewrite ---------------------------------------------------------
echo "Rewriting history..."
git filter-repo --force \
  --replace-text "$REPLACEMENTS" \
  --replace-message "$REPLACEMENTS"

# filter-repo deliberately drops remotes so a rewrite cannot be pushed by
# accident. Restore it explicitly.
if [[ -n "$ORIGIN" ]] && ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$ORIGIN"
  echo "Restored origin -> $ORIGIN"
fi

# --- verify ----------------------------------------------------------
echo
echo "Verifying no sensitive strings remain anywhere in history..."
# Local branches only: refs/remotes still point at the pre-rewrite
# history until the push lands, and would give false failures.
ALL_COMMITS="$(git rev-list --branches)"
FAILED=0
for term in "$T_BANK" "$T_RAM" "$T_OS" "$T_JUP1" "$T_JUP2" "$T_JUP3"; do
  # `|| true`: git grep exits 1 on no match, which under `set -o pipefail`
  # would abort the script exactly when verification is succeeding.
  hits=$( { git grep -I -i -l -- "$term" $ALL_COMMITS 2>/dev/null || true; } | wc -l | tr -d ' ')
  if [[ "$hits" != "0" ]]; then
    echo "  FAIL: '$term' still present in $hits file-revisions"
    { git grep -I -i -l -- "$term" $ALL_COMMITS 2>/dev/null || true; } | head -3 | sed 's/^/        /'
    FAILED=1
  else
    echo "  ok: '$term' gone"
  fi
done
msgs=$(git log --branches --format=%B | grep -ic -- "$T_BANK" || true)
if [[ "$msgs" != "0" ]]; then
  echo "  FAIL: bank name still in $msgs commit-message lines"
  FAILED=1
else
  echo "  ok: commit messages clean"
fi
[[ "$FAILED" == "0" ]] || { echo; echo "Verification FAILED -- not pushing."; exit 1; }

# --- push ------------------------------------------------------------
echo
if [[ "$PUSH" == "true" ]]; then
  echo "Force-pushing rewritten history to origin/$BRANCH..."
  git push --force origin "$BRANCH"
  git branch --set-upstream-to=origin/"$BRANCH" "$BRANCH" >/dev/null 2>&1 || true
  echo "Pushed."
else
  echo "Local history rewritten and verified. Not pushed."
  echo "  push with: git push --force origin $BRANCH"
fi
