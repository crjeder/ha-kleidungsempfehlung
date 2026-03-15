#!/bin/bash
# validate.sh - Full validation of kleidungsempfehlung in HA docker container
# Usage: HA_TOKEN=<long_lived_token> ./scripts/validate.sh
# Or:    ./scripts/validate.sh  (will prompt for token if not set)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HA_URL="http://localhost:8123"

# Token handling
if [ -z "$HA_TOKEN" ]; then
  if [ -f "$PROJECT_DIR/.ha_token" ]; then
    HA_TOKEN=$(cat "$PROJECT_DIR/.ha_token")
  else
    echo "ERROR: HA_TOKEN not set and .ha_token file not found."
    echo "Create a Long-Lived Access Token in HA (Profile → Security) and either:"
    echo "  export HA_TOKEN=<token>"
    echo "  echo <token> > $PROJECT_DIR/.ha_token"
    exit 1
  fi
fi

echo "======================================"
echo " Kleidungsempfehlung Validation"
echo "======================================"

# 1. Start container if not running
if ! docker ps --format '{{.Names}}' | grep -q "^ha-test$"; then
  echo "[1/4] Starting HA container..."
  cd "$PROJECT_DIR/ha-test"
  docker compose up -d
else
  echo "[1/4] Container already running"
fi

# 2. Wait for HA API to be ready
echo "[2/4] Waiting for HA to be ready..."
MAX_WAIT=120
ELAPSED=0
until curl -sf "$HA_URL/api/" -H "Authorization: Bearer $HA_TOKEN" > /dev/null 2>&1; do
  sleep 3
  ELAPSED=$((ELAPSED + 3))
  if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: HA did not become ready after ${MAX_WAIT}s"
    echo "--- Container logs (last 50 lines) ---"
    docker logs ha-test --tail 50
    exit 1
  fi
  echo "  ...waiting ($ELAPSED/${MAX_WAIT}s)"
done
echo "  HA is ready."

# 3. Check integration logs for errors
echo ""
echo "[3/4] Checking integration logs..."
echo "--- kleidungsempfehlung log output ---"
docker logs ha-test 2>&1 | grep -i "kleidungsempfehlung" || echo "  (no log lines found for integration)"
echo ""

ERROR_COUNT=$(docker logs ha-test 2>&1 | grep -i "kleidungsempfehlung" | grep -icE "error|exception|traceback" || true)
if [ "$ERROR_COUNT" -gt 0 ]; then
  echo "⚠️  Found $ERROR_COUNT error/exception line(s) in logs:"
  docker logs ha-test 2>&1 | grep -i "kleidungsempfehlung" | grep -iE "error|exception|traceback"
else
  echo "✅ No errors in integration logs"
fi

# 4. Validate sensor states
echo ""
echo "[4/4] Validating sensor states..."
python3 "$SCRIPT_DIR/check_sensors.py" "$HA_URL" "$HA_TOKEN"

echo ""
echo "======================================"
echo " Validation complete."
echo "======================================"
