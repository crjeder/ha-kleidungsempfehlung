#!/bin/bash
# restart.sh - Restart HA container and tail integration logs
# Use after code changes to reload the integration.

echo "Restarting ha-test container..."
docker restart ha-test

echo "Waiting for startup..."
sleep 5

echo "--- Tailing kleidungsempfehlung log output (Ctrl+C to stop) ---"
docker logs ha-test -f 2>&1 | grep --line-buffered -iE "kleidungsempfehlung|error|exception|traceback"
