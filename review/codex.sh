#!/usr/bin/env bash
# An independent Codex review of a frozen copy of the repo.
#
#   review/codex.sh <lens> [focus...]     lens: code | research | usability | results (review/lenses/*.md)
#   review/codex.sh code "Review only loud/identity.py and its configs."
#
# Writes review/reports/<stamp>-<lens>-codex.md. Codex's own sandbox cannot start in this pod (bubblewrap needs
# mount privileges the container lacks), so the isolation is an OS user: Codex runs as loud-auditor, which can read
# everything here but write only its work directory under /var/tmp/loud-review. Run as root.
set -euo pipefail

lens=${1:?usage: review/codex.sh <lens> [focus...]}
shift
focus=${*:-}
repo=$(cd "$(dirname "$0")/.." && pwd)
[[ -f $repo/review/lenses/$lens.md ]] || { echo "no lens '$lens'; have: $(ls "$repo/review/lenses")" >&2; exit 2; }

user=loud-auditor
home=/home/$user
stamp=$(date -u +%Y%m%dT%H%M%SZ)
work=/var/tmp/loud-review/$stamp-$lens
report=$repo/review/reports/$stamp-$lens-codex.md
id "$user" >/dev/null 2>&1 || useradd --system --create-home --home-dir "$home" --shell /bin/bash "$user"

# Codex refreshes its login by rotating the token, so whichever copy refreshed last wins in both places.
sync_login() {
    python3 - "$home/.codex/auth.json" <<'EOF'
import json, os, pwd, shutil, sys
from pathlib import Path
mine, theirs = Path('/root/.codex/auth.json'), Path(sys.argv[1])
stamp = lambda p: json.loads(p.read_text()).get('last_refresh', '') if p.exists() else ''
src, dst = (theirs, mine) if stamp(theirs) > stamp(mine) else (mine, theirs)
if stamp(src) != stamp(dst):
    tmp = dst.with_suffix('.tmp')
    shutil.copyfile(src, tmp)
    owner = 0 if dst == mine else pwd.getpwnam('loud-auditor').pw_uid
    os.chown(tmp, owner, owner)
    os.chmod(tmp, 0o600)
    tmp.replace(dst)
EOF
}
install -d -o "$user" -g "$user" -m 700 "$home/.codex"
sync_login
trap sync_login EXIT

# freeze the source under review (tracked and untracked, minus .gitignore)
mkdir -p "$work/src" "$work/scratch"
git -C "$repo" ls-files -co --exclude-standard -z | tar -C "$repo" --null -T - -cf - | tar -C "$work/src" -xf -
digest=$(cd "$work/src" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
{
    cat "$repo/review/RUBRIC.md"
    echo
    cat "$repo/review/lenses/$lens.md"
    echo
    echo "## This review"
    echo
    echo "- Source under review (a frozen copy, your working directory): $work/src"
    echo "- The live repo, its data, pinned external code and runs (read-only): $repo"
    echo "- Your scratch directory (the only place you may write): $work/scratch"
    [[ -n $focus ]] && echo "- Focus: $focus"
    echo "- Give paths in findings relative to the repo root."
} > "$work/prompt.md"
chown -R "$user:$user" "$work"

echo "codex $lens review of $digest running; log: $work/codex.log" >&2
runuser -u "$user" -- env -i HOME="$home" USER="$user" CODEX_HOME="$home/.codex" PATH=/usr/local/bin:/usr/bin:/bin \
    LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 LOUD_REPO="$repo" TMPDIR="$work/scratch" \
    codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C "$work/src" \
    -o "$work/report.md" - < "$work/prompt.md" > "$work/codex.log" 2>&1 || status=$?

model=$(grep -m1 '^model:' "$work/codex.log" | cut -d' ' -f2- || true)
session=$(grep -m1 '^session id:' "$work/codex.log" | cut -d' ' -f3 || true)
{
    echo "# $lens review: Codex"
    echo
    echo "- source: frozen copy \`$digest\` at $stamp (git $(git -C "$repo" rev-parse --short HEAD) plus the working tree)"
    echo "- reviewer: Codex $(codex --version | cut -d' ' -f2), model $model, session $session"
    [[ -n $focus ]] && echo "- focus: $focus"
    echo
    if [[ -s $work/report.md ]]; then cat "$work/report.md"; else echo "**No report.** See \`$work/codex.log\`."; fi
} > "$report"
echo "$report"
exit "${status:-0}"
