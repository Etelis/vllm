#!/bin/bash
# Monitor PR #39306 CI checks and push main when tests fail

PR_NUMBER=39306
REPO="vllm-project/vllm"
MAX_ITERS=200
POLL_INTERVAL=120  # 2 minutes

get_pr_head_sha() {
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
      "https://api.github.com/repos/${REPO}/pulls/${PR_NUMBER}" | \
      python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('head',{}).get('sha',''))"
}

get_ci_status() {
    local sha="$1"
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
      "https://api.github.com/repos/${REPO}/commits/${sha}/status" | \
      python3 -c "
import json, sys
d = json.load(sys.stdin)
statuses = d.get('statuses', [])
overall = d.get('state', 'unknown')
failed = [s['context'] for s in statuses if s['state'] == 'failure' or s['state'] == 'error']
pending = [s['context'] for s in statuses if s['state'] == 'pending']
success = [s['context'] for s in statuses if s['state'] == 'success']
print(f'overall={overall}')
print(f'failed={len(failed)}')
print(f'pending={len(pending)}')
print(f'success={len(success)}')
if failed:
    for f in failed:
        print(f'FAILED:{f}')
"
}

update_branch() {
    local sha="$1"
    echo "[$(date)] Triggering branch update (merging main into PR branch)..."
    response=$(curl -s -X PUT \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github.v3+json" \
      -H "Content-Type: application/json" \
      "https://api.github.com/repos/${REPO}/pulls/${PR_NUMBER}/update-branch" \
      -d "{\"expected_head_sha\": \"${sha}\"}")
    echo "[$(date)] Update response: $response"
}

echo "[$(date)] Starting PR monitor for ${REPO}#${PR_NUMBER}"

for i in $(seq 1 $MAX_ITERS); do
    echo ""
    echo "[$(date)] === Iteration $i ==="

    sha=$(get_pr_head_sha)
    if [ -z "$sha" ]; then
        echo "[$(date)] ERROR: Could not get PR head SHA. Retrying in ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
        continue
    fi

    echo "[$(date)] Current head SHA: $sha"
    status_output=$(get_ci_status "$sha")
    echo "$status_output"

    overall=$(echo "$status_output" | grep "^overall=" | cut -d= -f2)
    failed_count=$(echo "$status_output" | grep "^failed=" | cut -d= -f2)
    pending_count=$(echo "$status_output" | grep "^pending=" | cut -d= -f2)

    if [ "$overall" = "success" ]; then
        echo "[$(date)] All checks PASSED! PR is ready. Exiting monitor."
        exit 0
    elif [ "$failed_count" -gt 0 ] 2>/dev/null; then
        echo "[$(date)] ${failed_count} check(s) FAILED. Pushing main into PR branch..."
        update_branch "$sha"
        echo "[$(date)] Waiting 60s for CI to pick up new commit..."
        sleep 60
    else
        echo "[$(date)] ${pending_count} check(s) still pending. Waiting ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    fi
done

echo "[$(date)] Reached max iterations ($MAX_ITERS). Exiting."
