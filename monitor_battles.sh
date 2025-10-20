#!/bin/bash
# Monitor ongoing battles between Vanilla MCTS and Abra MCTS

echo "Monitoring Pokemon Showdown battles..."
echo "Vanilla MCTS vs Abra MCTS (with Minikazam neural guidance)"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "BATTLE MONITOR"
    echo "=========================================="
    echo ""

    echo "--- Vanilla MCTS Status ---"
    if [ -f vanilla_mcts.log ]; then
        echo "Last 5 lines:"
        tail -5 vanilla_mcts.log | grep -E "INFO|ERROR|WARNING|Turn|Battle|Won|Lost" || echo "  (searching for battles...)"
    else
        echo "  No log file yet"
    fi
    echo ""

    echo "--- Abra MCTS Status (Minikazam) ---"
    if [ -f abra_mcts.log ]; then
        echo "Last 5 lines:"
        tail -5 abra_mcts.log | grep -E "INFO|ERROR|WARNING|Turn|Battle|Won|Lost" || echo "  (searching for battles...)"
    else
        echo "  No log file yet"
    fi
    echo ""

    echo "--- Battle Results ---"
    echo "Vanilla MCTS wins: $(grep -c "won the battle" vanilla_mcts.log 2>/dev/null || echo 0)"
    echo "Abra MCTS wins: $(grep -c "won the battle" abra_mcts.log 2>/dev/null || echo 0)"
    echo ""

    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
