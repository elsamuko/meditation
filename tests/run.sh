#!/usr/bin/env bash
# Run the DSP tests in headless Chrome. Exit code 0 = all passed.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
[ -x "$CHROME" ] || CHROME="$(command -v google-chrome || command -v chromium || true)"
[ -n "$CHROME" ] || { echo "Chrome not found; set CHROME=/path/to/chrome or open tests/index.html via a local server"; exit 2; }

OUT="$("$CHROME" --headless=new --disable-gpu --allow-file-access-from-files \
  --virtual-time-budget=120000 --dump-dom "file://$DIR/index.html" 2>/dev/null)"

# print the <pre id="out"> contents, unescaped
echo "$OUT" | python3 -c '
import sys, re, html
m = re.search(r"<pre id=\"out\">(.*?)</pre>", sys.stdin.read(), re.S)
print(html.unescape(m.group(1)) if m else "no test output (module failed to load?)")'

echo "$OUT" | grep -q "PASS: " && ! echo "$OUT" | grep -q "FAIL: "
