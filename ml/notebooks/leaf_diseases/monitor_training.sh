#!/bin/bash
# Training Monitor Script

LOG_FILE="training_log.txt"
LAST_LINE=0

echo "======================================"
echo "  Training Monitor Started"
echo "======================================"
echo ""

while true; do
    if [ -f "$LOG_FILE" ]; then
        # Get current line count
        CURRENT_LINES=$(wc -l < "$LOG_FILE")

        if [ "$CURRENT_LINES" -gt "$LAST_LINE" ]; then
            # Show new lines since last check
            NEW_CONTENT=$(tail -n +$((LAST_LINE + 1)) "$LOG_FILE" | grep -E "Epoch [0-9]+/[0-9]+|PHASE|Test Accuracy|completed in|Best validation")

            if [ ! -z "$NEW_CONTENT" ]; then
                echo "----------------------------------------"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
                echo "$NEW_CONTENT"
                echo "----------------------------------------"
                echo ""
            fi

            LAST_LINE=$CURRENT_LINES
        fi
    fi

    # Check every 2 minutes
    sleep 120
done
