#!/usr/bin/env python3
"""Run two agents against each other N times and report win counts."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import glob

NUM_GAMES = 10
PARALLEL_GAMES = 10
MATCH_DIR = "3600-agents/matches"

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <agent_a> <agent_b>")
    sys.exit(1)

agent_a = sys.argv[1]
agent_b = sys.argv[2]

print(f"Running {agent_a} vs {agent_b} for {NUM_GAMES} games...\n")

# Clean old match files for these agents so we only pick up new ones
for f in glob.glob(os.path.join(MATCH_DIR, "*.json")):
    try:
        with open(f) as fh:
            data = json.load(fh)
        if data.get("player_a") == agent_a and data.get("player_b") == agent_b:
            os.remove(f)
    except Exception:
        pass

wins_a = 0
wins_b = 0
ties = 0


def _extract_match_path(output: str) -> str | None:
    marker = "MATCH_FILE:"
    for line in output.splitlines():
        if line.startswith(marker):
            return line[len(marker) :].strip()
    return None


def _run_one_game(game_idx: int):
    del game_idx
    result = subprocess.run(
        ["python3", "engine/run_local_agents.py", agent_a, agent_b],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return "error", result.stderr.strip() if result.stderr else None

    match_path = _extract_match_path(result.stdout)
    if not match_path or not os.path.exists(match_path):
        return "missing", None

    try:
        with open(match_path) as fh:
            data = json.load(fh)
    except Exception:
        return "parse_error", match_path
    finally:
        try:
            os.remove(match_path)
        except OSError:
            pass

    return "ok", data.get("result", -1)


done = 0
with ThreadPoolExecutor(max_workers=PARALLEL_GAMES) as pool:
    futures = [pool.submit(_run_one_game, i) for i in range(NUM_GAMES)]
    for future in as_completed(futures):
        status, payload = future.result()
        done += 1

        if status == "ok":
            if payload == 0:
                wins_a += 1
            elif payload == 1:
                wins_b += 1
            else:
                ties += 1
        elif status == "error":
            print(f"  Game {done}: process failed")
            if payload:
                print(f"    {payload}")
            ties += 1
        elif status == "parse_error":
            print(f"  Game {done}: failed to parse {payload}")
            ties += 1
        else:
            print(f"  Game {done}: match file missing")
            ties += 1

        if done % 5 == 0 or done == NUM_GAMES:
            print(
                f"  Games {done}/{NUM_GAMES} done  |  {agent_a}: {wins_a}  {agent_b}: {wins_b}  Ties: {ties}"
            )

print(f"\n{'=' * 50}")
print(f"Final: {agent_a}: {wins_a} wins  |  {agent_b}: {wins_b} wins  |  Ties: {ties}")
if wins_a > wins_b:
    print(f"Winner: {agent_a}")
elif wins_b > wins_a:
    print(f"Winner: {agent_b}")
else:
    print("Result: Tie")
print(f"{'=' * 50}")
