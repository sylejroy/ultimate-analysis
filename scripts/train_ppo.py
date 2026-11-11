#!/usr/bin/env python3
"""Train simulated teams by reinforcement (PPO) from copied players (simulation/ppo.py).

Each iteration the present team plays points, both sides of a point when it plays
itself, against the scripted team or an earlier version of itself otherwise, a random
wind each point, at most --point-seconds each. Every player's every step is kept: what
they saw, what they did, how likely that was, and the value they saw in it. Rewards
come from what happened (ppo.rewards_of: completions and the yards each gained,
turnovers, scores; the other team gets the opposite; the defence also gains
--stall-reward a second its mark counts the stall), shared by the team; each step's
advantage is how much better it went than its value said (GAE). The policy then moves
towards what went better, never far from what played (clipped, PPO).

Progress is printed per iteration: points scored and conceded, completions, the reward
per point; every --report iterations the team plays the scripted team and the copied
team it started from, as it would without trying things, and is kept in a new folder:

    <output>/<run>/iter_0010/ppo.pt, minds.pt ... and latest/

    python scripts/train_ppo.py data/models/movement/20261107_235856_imitation \\
        --iterations 300 --workers 12
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.config.settings import get_setting  # noqa: E402
from ultimate_analysis.simulation import game  # noqa: E402
from ultimate_analysis.simulation.learned import (  # noqa: E402
    LearnedTeam,
    keep_wiring,
    load_minds,
    rewire,
    wiring_of,
)
from ultimate_analysis.simulation.match import play_point  # noqa: E402
from ultimate_analysis.simulation.ppo import (  # noqa: E402
    STALL_REWARD,
    Brains,
    PolicyTeam,
    _log_chance,
    load_brains,
    player_rewards,
    rewards_of,
    save_brains,
    stall_watch,
)
from ultimate_analysis.simulation.scripted import ScriptedTeam  # noqa: E402

GAMMA, LAMBDA = 0.99, 0.95  # Per tenth of a second: rewards count for some 10 s on
CLIP = 0.2
EPOCHS, BATCH = 3, 8192
ENTROPY = 0.001
SELF, SCRIPTED = 0.6, 0.2  # Shares of the points against itself and the scripted team

# ---------------------------------------------------------------------- playing (workers)

_start = None
_throw_net = None


def _start_worker(start_folder: str, throw_weights: str) -> None:
    global _start, _throw_net
    from ultimate_analysis.processing.throw_targets import load_throw_targets

    torch.set_num_threads(1)
    _start = load_minds(Path(start_folder))
    _throw_net = load_throw_targets(throw_weights)


def _brains(state):
    brains = Brains.from_minds(_start)
    brains.load(state)
    return brains


def play(job) -> list:
    """Play points with the present team (its state) against an opponent: "self",
    "scripted", "start" (the copied team, as it is) or the state of an earlier team.
    Per point: the samples of the learning team's players, its rewards, and what came of
    it."""
    state, opponent, seeds, limit, greedy, stall_reward = job
    found = []
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        side = k % 2  # The learning team is team 0 or team 1 by turns
        ours = PolicyTeam(_brains(state), side, rng, record=not greedy, greedy=greedy)
        if opponent == "self":
            theirs = PolicyTeam(ours.brains, 1 - side, rng, record=not greedy, greedy=greedy)
        elif opponent == "scripted":
            theirs = ScriptedTeam(seed)
        elif opponent == "start":
            theirs = LearnedTeam(_start, _throw_net, seed)
        else:
            theirs = PolicyTeam(_brains(opponent), 1 - side, rng, record=False, greedy=greedy)
        teams = (ours, theirs) if side == 0 else (theirs, ours)
        # Who receives the pull by a fair coin: by turns would favour one side when a
        # worker plays an odd number of points
        start = game.pull_start(rng, int(rng.integers(2)), game.a_wind(rng))
        counted, watch = stall_watch()
        point = play_point(teams, rng, start, watch=watch, limit=limit)
        steps = max(ours.step, 1)
        rewards = rewards_of(point, steps, game.STEP, counted, stall_reward)
        own = player_rewards(point, steps, game.STEP)
        kinds = [e.kind for e in point.events]
        samples = ours.samples + (
            theirs.samples if isinstance(theirs, PolicyTeam) and theirs.record else []
        )
        found.append(
            {
                "samples": samples,
                "rewards": rewards,
                "own": own,
                "side": side,
                "scored": int(point.scored == side),
                "conceded": int(point.scored == 1 - side),
                "throws": kinds.count("throw"),
                "catches": kinds.count("catch"),
                "reward": float(rewards[side].sum()),
                "stalled": counted.count(1 - side) * game.STEP,
            }
        )
    return found


# ---------------------------------------------------------------------- learning


def advantages(points: list, credit: float = 1.0) -> list:
    """Each sample with its advantage and return (GAE over each player's steps), and what
    the team really got from then on (`happened`: the rewards to the end of the point,
    discounted as the values are). Each
    player's reward: their team's, and `credit` times their own (ppo.player_rewards)."""
    kept = []
    for p, point in enumerate(points):
        by_player = defaultdict(list)
        for sample in point["samples"]:
            by_player[(sample.team, sample.player)].append(sample)
        for (team, player), samples in by_player.items():
            samples.sort(key=lambda s: s.step)
            rewards = point["rewards"][team] + credit * point["own"][player + team * game.PLAYERS]
            running = happened = 0.0
            for i in range(len(samples) - 1, -1, -1):
                s = samples[i]
                after = samples[i + 1].value if i + 1 < len(samples) else 0.0
                reward = rewards[min(s.step, len(rewards) - 1)]
                delta = reward + GAMMA * after - s.value
                running = delta + GAMMA * LAMBDA * running
                happened = reward + GAMMA * happened
                s.advantage, s.returned, s.happened = running, running + s.value, happened
                kept.append(s)
    return kept


def learn(
    brains: Brains,
    samples: list,
    optimisers: dict,
    wiring: dict,
    value_optimisers: dict,
    values_only: bool = False,
    value_target: str = "happened",
) -> dict:
    """PPO steps on the samples of each side; cut connections stay cut. The values learn
    by their own optimisers, towards what happened (`value_target` "happened") or the
    GAE return ("gae": partly their own guess of what follows); with `values_only`
    nothing else changes."""
    said = {}
    for side in ("offence", "defence"):
        policy = getattr(brains, side)
        mine = [s for s in samples if s.side == side]
        if not mine:
            continue
        x = torch.from_numpy(np.stack([s.inputs for s in mine]))
        seen = torch.from_numpy(np.stack([s.value_inputs for s in mine]))
        action = torch.from_numpy(np.stack([s.action for s in mine]))
        old = torch.tensor([s.log_chance for s in mine])
        moved = torch.tensor([s.moved for s in mine])
        advantage = torch.tensor([s.advantage for s in mine], dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        returned = torch.tensor(
            [s.happened if value_target == "happened" else s.returned for s in mine],
            dtype=torch.float32,
        )
        holders = [i for i, s in enumerate(mine) if s.throw_inputs is not None]
        losses = []
        for _ in range(EPOCHS):
            order = torch.randperm(len(mine))
            for start in range(0, len(order), BATCH):
                batch = order[start : start + BATCH]
                spread = torch.exp(policy.log_spread)
                mean = policy.movement(x[batch])
                ratio = torch.exp(_log_chance(action[batch], mean, spread) - old[batch])
                a = advantage[batch]
                clipped = torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a)
                move_loss = -(clipped * moved[batch]).sum() / moved[batch].sum().clamp(min=1)
                value_loss = ((policy.value(seen[batch])[:, 0] - returned[batch]) ** 2).mean()
                entropy = policy.log_spread.sum()
                loss = 0.5 * value_loss
                if not values_only:
                    loss = loss + move_loss - ENTROPY * entropy
                optimisers[side].zero_grad()
                value_optimisers[side].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                if not values_only:
                    optimisers[side].step()
                value_optimisers[side].step()
                keep_wiring(policy.movement, wiring[side])
                losses.append(float(loss))
            if side == "offence" and holders and not values_only:
                loss = _holder_loss(policy, [mine[i] for i in holders], advantage[holders])
                optimisers[side].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimisers[side].step()
        said[side] = float(np.mean(losses))
    return said


def _holder_loss(policy, samples: list, advantage: torch.Tensor) -> torch.Tensor:
    """The clipped loss of the holder's choices: whether to throw now, and where to."""
    throw = torch.from_numpy(np.stack([s.throw_inputs for s in samples]))
    threw = torch.tensor([s.threw for s in samples], dtype=torch.float32)
    logit = policy.throw_now(throw)[:, 0]
    new = threw * torch.nn.functional.logsigmoid(logit) + (
        1 - threw
    ) * torch.nn.functional.logsigmoid(-logit)
    thrown = [i for i, s in enumerate(samples) if s.threw]
    if thrown:
        squares = torch.from_numpy(
            np.stack([samples[i].square_inputs.astype(np.float32) for i in thrown])
        )
        prior = torch.from_numpy(np.stack([samples[i].prior for i in thrown]))
        chosen = torch.tensor([samples[i].square for i in thrown])
        logits = policy.squares(squares)[..., 0] + prior
        where = torch.log_softmax(logits, dim=1)[torch.arange(len(thrown)), chosen]
        new = new.index_add(0, torch.tensor(thrown), where)
    old = torch.tensor([s.throw_log_chance for s in samples])
    ratio = torch.exp(new - old)
    return -torch.min(ratio * advantage, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage).mean()


# ---------------------------------------------------------------------- the loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "start",
        type=Path,
        help="Folder of the minds to start from, or of an earlier run's team (with ppo.pt) to go on with",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--points", type=int, default=36, help="Per iteration")
    parser.add_argument("--point-seconds", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument(
        "--value-rate", type=float, default=1e-3, help="The value networks' learning rate"
    )
    parser.add_argument(
        "--credit",
        type=float,
        default=1.0,
        help="How much each player's own reward (what they did) counts beside the team's",
    )
    parser.add_argument(
        "--value-target",
        choices=("happened", "gae"),
        default="happened",
        help="What the values learn: the rewards that followed to the end of the point, or "
        "the GAE return (which leans on the values' own guesses)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Iterations at first in which only the values learn (the players as they are)",
    )
    parser.add_argument(
        "--message-spread",
        type=float,
        default=None,
        help="How widely what players say is tried around what they mean to say, at first",
    )
    parser.add_argument(
        "--stall-reward",
        type=float,
        default=STALL_REWARD,
        help="To the defence per second its mark counts the stall",
    )
    parser.add_argument("--report", type=int, default=10)
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Rewiring thins the networks to this share of their connections",
    )
    parser.add_argument(
        "--rewire-every", type=int, default=0, help="Iterations between rewirings; 0: never"
    )
    parser.add_argument(
        "--self-share", type=float, default=SELF, help="Share of the points against itself"
    )
    parser.add_argument(
        "--scripted-share",
        type=float,
        default=SCRIPTED,
        help="Share of the points against the scripted team; the rest against earlier versions",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/models/movement"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # Going on from an earlier run takes its throwing and value networks along too
    resumed = (args.start / "ppo.pt").exists()
    brains = load_brains(args.start) if resumed else Brains.from_minds(load_minds(args.start))
    if args.message_spread is not None:
        for side in ("offence", "defence"):
            with torch.no_grad():
                getattr(brains, side).log_spread[2:] = float(np.log(args.message_spread))

    def acting(policy):
        values = {id(p) for p in policy.value.parameters()}
        return [p for p in policy.parameters() if id(p) not in values]

    optimisers = {
        side: torch.optim.Adam(acting(getattr(brains, side)), lr=args.rate)
        for side in ("offence", "defence")
    }
    value_optimisers = {
        side: torch.optim.Adam(getattr(brains, side).value.parameters(), lr=args.value_rate)
        for side in ("offence", "defence")
    }
    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_ppo"
    run.mkdir(parents=True)
    settings = {**vars(args), "start": str(args.start), "output": str(args.output)}
    (run / "settings.json").write_text(json.dumps(settings, indent=2))
    throw_weights = str(get_setting("models.throw_targets.weights"))
    league = []
    wiring = {side: wiring_of(getattr(brains, side).movement) for side in ("offence", "defence")}
    print(f"Written to {run}", flush=True)
    with Pool(args.workers, _start_worker, (str(args.start), throw_weights)) as pool:
        for iteration in range(1, args.iterations + 1):
            began = time.perf_counter()
            state = brains.state()
            jobs = []
            for _ in range(args.workers):
                share = args.points // args.workers or 1
                draw = rng.random()
                # Before there are earlier versions, their share goes to the scripted team
                opponent = (
                    "self"
                    if draw < args.self_share
                    else "scripted"
                    if draw < args.self_share + args.scripted_share or not league
                    else league[int(rng.integers(len(league)))]
                )
                seeds = [int(s) for s in rng.integers(0, 2**31, share)]
                jobs.append((state, opponent, seeds, args.point_seconds, False, args.stall_reward))
            points = [p for found in pool.map(play, jobs) for p in found]
            played = time.perf_counter() - began
            samples = advantages(points, args.credit)
            if args.rewire_every and iteration % args.rewire_every == 0:
                for side in ("offence", "defence"):
                    wiring[side] = rewire(getattr(brains, side).movement, rng, density=args.density)
            said = learn(
                brains,
                samples,
                optimisers,
                wiring,
                value_optimisers,
                values_only=iteration <= args.warmup,
                value_target=args.value_target,
            )
            throws = sum(p["throws"] for p in points)
            print(
                f"iteration {iteration}: {len(points)} points, scored {sum(p['scored'] for p in points)}, "
                f"conceded {sum(p['conceded'] for p in points)}, completed "
                f"{sum(p['catches'] for p in points)}/{throws}, reward per point "
                f"{np.mean([p['reward'] for p in points]):+.2f}, the other team's stall counted "
                f"{np.mean([p['stalled'] for p in points]):.1f} s a point, {len(samples)} steps, "
                f"spread {torch.exp(brains.offence.log_spread[:2]).mean():.3f}, "
                f"{played:.0f} s playing, {time.perf_counter() - began - played:.0f} s learning, "
                f"loss {said.get('offence', 0):.3f} offence, {said.get('defence', 0):.3f} defence"
                + (" (values only)" if iteration <= args.warmup else ""),
                flush=True,
            )
            if iteration % args.report == 0:
                folder = run / f"iter_{iteration:04d}"
                save_brains(brains, folder, iteration=iteration)
                save_brains(brains, run / "latest", iteration=iteration)
                league.append(brains.state())
                said = []
                for name in ("scripted", "start"):
                    seeds = [int(s) for s in rng.integers(0, 2**31, 16)]
                    found = pool.apply(play, ((brains.state(), name, seeds, 120.0, True, 0.0),))
                    scored = sum(p["scored"] for p in found)
                    conceded = sum(p["conceded"] for p in found)
                    said.append(
                        f"against {name}: scored {scored}, conceded {conceded} of 16 points"
                    )
                    (folder / f"against_{name}.json").write_text(
                        json.dumps({"scored": scored, "conceded": conceded})
                    )
                print("  " + "; ".join(said), flush=True)


if __name__ == "__main__":
    main()
