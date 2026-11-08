#!/usr/bin/env python3
"""Copy the scripted team into a team that learns (simulation/ppo.py), to go on with by
PPO (scripts/train_ppo.py).

The scripted team plays itself (pulls, a random wind each point, no set plays: a play
called is heard as a message the learned team does not have). Kept per step, where
neither the holder nor a reflex decided: what each player saw (learned.py's inputs,
with nothing heard: the copy says nothing at first) and how they ran, as the change the
movement network gives. Kept per holder step: what the holder's choices see
(ppo.holder_inputs), whether the scripted thrower threw, and on a throw which teammate
they led. Kept for the values: each player's return from then on (ppo.rewards_of, with
the defence gaining while its mark counts the stall).

The networks then learn these, printed per epoch on the points kept apart: how players
run (mean squared error of the change, what they say towards nothing), when the holder
throws (log loss) and to which choice (log loss, and how often the scripted thrower's
choice is the likeliest), and the values. Written to a new folder, as train_ppo.py
writes a team (ppo.pt and minds.pt), then played against the scripted team:

    python scripts/imitate_scripted.py data/models/movement/20261107_235856_imitation \\
        --points 400 --workers 12
"""

import argparse
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.simulation import game  # noqa: E402
from ultimate_analysis.simulation.learned import (  # noqa: E402
    INPUT_GROUPS,
    load_minds,
    reflexes,
    view_features,
)
from ultimate_analysis.simulation.match import Match  # noqa: E402
from ultimate_analysis.simulation.ppo import (  # noqa: E402
    STALL_REWARD,
    Brains,
    PolicyTeam,
    critic_inputs,
    holder_inputs,
    rewards_of,
    save_brains,
    stall_watch,
)
from ultimate_analysis.simulation.scripted import ScriptedTeam  # noqa: E402

GAMMA = 0.99  # As train_ppo.py
EVERY = 2  # Steps between those kept of how players run and their values
HEARD = slice(
    sum(n for name, n in INPUT_GROUPS[: [g[0] for g in INPUT_GROUPS].index("Heard")]),
    sum(n for name, n in INPUT_GROUPS[: [g[0] for g in INPUT_GROUPS].index("Heard") + 1]),
)

# ---------------------------------------------------------------------- playing (workers)


def record(job) -> dict:
    """Play scripted points and keep what the copy learns from."""
    seeds, limit, stall_reward = job
    kept = {k: [] for k in ("x", "run", "side", "value_x", "returned", "value_side")}
    kept.update(throw=[], threw=[], choices=[], prior=[], choice=[])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        teams = (ScriptedTeam(seed, plays=False), ScriptedTeam(seed + 1, plays=False))
        start = game.pull_start(rng, int(rng.integers(2)), game.a_wind(rng))
        match = Match(teams, rng, start)
        seen = []  # Per step: (team, k, x, attacking) for the values
        counted, watch = stall_watch()
        step = 0
        while not match.state.over and match.state.seconds < limit:
            critic = [critic_inputs(match.state, team) for team in (0, 1)]  # Before the step
            match.step()
            watch(match.state)
            for team in (0, 1):
                views, decision = match.views[team], match.decisions[team]
                attacking = views[0].attacking
                x = np.stack([view_features(v) for v in views])
                x[:, HEARD] = 0.0
                decided = np.ones(game.PLAYERS, dtype=bool)
                holder = views[0].holder if attacking else None
                if holder is not None:
                    decided[holder] = False
                    _holder(views[holder], teams[team], kept)
                elif views[0].disc is not None:
                    for k in reflexes(views):
                        decided[k] = False
                if step % EVERY == 0:
                    velocities = np.asarray(decision.velocities, dtype=np.float32)
                    run = velocities / game.RUN_SPEED - x[:, 2:4]
                    for k in np.flatnonzero(decided):
                        kept["x"].append(x[k])
                        kept["run"].append(run[k])
                        kept["side"].append(attacking)
                    seen.extend(
                        (team, k, critic[team][k], attacking, step) for k in range(game.PLAYERS)
                    )
            step += 1
        rewards = rewards_of(match.state, max(step, 1), game.STEP, counted, stall_reward)
        returns = np.zeros_like(rewards)
        running = np.zeros(2)
        for i in range(rewards.shape[1] - 1, -1, -1):
            running = rewards[:, i] + GAMMA * running
            returns[:, i] = running
        for team, k, x, attacking, at in seen:
            kept["value_x"].append(x)
            kept["returned"].append(returns[team, at])
            kept["value_side"].append(attacking)
    return {
        k: np.array(v, dtype=np.float16 if k == "choices" else None) if len(v) else np.zeros(0)
        for k, v in kept.items()
    }


def _holder(view, team: ScriptedTeam, kept: dict) -> None:
    """The holder's choice as the scripted thrower made it (its throwing is a function of
    what it sees alone)."""
    if view.held < 0.3:
        return
    choices, prior, throw, targets = holder_inputs(view)
    target = team._throw(view)
    kept["throw"].append(throw)
    kept["threw"].append(target is not None)
    if target is not None:
        led = targets[-(game.PLAYERS - 1) :]
        kept["choices"].append(choices)
        kept["prior"].append(prior)
        choice = len(targets) - len(led) + int(np.argmin(np.hypot(*(led - target).T)))
        kept["choice"].append(choice)


# ---------------------------------------------------------------------- learning


def _batches(n: int, size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for start in range(0, n, size):
        yield torch.from_numpy(order[start : start + size])


def learn_running(policy, x, run, held_x, held_run, epochs, rng, name):
    optimiser = torch.optim.Adam(policy.movement.parameters(), lr=1e-3)
    x, run = torch.from_numpy(x), torch.from_numpy(run)
    held_x, held_run = torch.from_numpy(held_x), torch.from_numpy(held_run)

    def loss_of(xs, runs):
        out = policy.movement(xs)
        return ((out[:, :2] - runs) ** 2).sum(1).mean(), (out[:, 2:] ** 2).sum(1).mean()

    for epoch in range(1, epochs + 1):
        for batch in _batches(len(x), 4096, rng):
            moved, said = loss_of(x[batch], run[batch])
            optimiser.zero_grad()
            (moved + 0.1 * said).backward()
            optimiser.step()
        with torch.no_grad():
            moved, said = loss_of(held_x, held_run)
            still = (held_run**2).sum(1).mean()
        print(
            f"  {name} running, epoch {epoch}: error {float(moved):.4f} on the points kept "
            f"apart (keeping on as they ran: {float(still):.4f}); said {float(said):.4f}",
            flush=True,
        )


def learn_values(policy, x, returned, held_x, held_returned, epochs, rng, name):
    optimiser = torch.optim.Adam(policy.value.parameters(), lr=1e-3)
    x, returned = torch.from_numpy(x), torch.from_numpy(returned).float()
    held_x, held_returned = torch.from_numpy(held_x), torch.from_numpy(held_returned).float()
    for epoch in range(1, epochs + 1):
        for batch in _batches(len(x), 4096, rng):
            loss = ((policy.value(x[batch])[:, 0] - returned[batch]) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            error = ((policy.value(held_x)[:, 0] - held_returned) ** 2).mean()
        spread = float(held_returned.var())
        print(
            f"  {name} values, epoch {epoch}: squared error {float(error):.3f} "
            f"(of a spread of {spread:.3f})",
            flush=True,
        )


def learn_throwing(policy, data, held, epochs, rng):
    optimiser = torch.optim.Adam(
        list(policy.throw_now.parameters()) + list(policy.squares.parameters()), lr=1e-3
    )

    def tensors(d):
        return (
            torch.from_numpy(d["throw"].astype(np.float32)),
            torch.from_numpy(d["threw"].astype(np.float32)),
            torch.from_numpy(d["choices"].astype(np.float32)),
            torch.from_numpy(d["prior"].astype(np.float32)),
            torch.from_numpy(d["choice"].astype(np.int64)),
        )

    throw, threw, choices, prior, choice = tensors(data)
    h_throw, h_threw, h_choices, h_prior, h_choice = tensors(held)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for epoch in range(1, epochs + 1):
        for batch in _batches(len(throw), 2048, rng):
            loss = bce(policy.throw_now(throw[batch])[:, 0], threw[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        for batch in _batches(len(choices), 256, rng):
            logits = policy.squares(choices[batch])[..., 0] + prior[batch]
            loss = torch.nn.functional.cross_entropy(logits, choice[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            when = float(bce(policy.throw_now(h_throw)[:, 0], h_threw))
            base = float(h_threw.mean())
            floor = -(base * np.log(base) + (1 - base) * np.log(1 - base))
            logits = policy.squares(h_choices)[..., 0] + h_prior
            where = float(torch.nn.functional.cross_entropy(logits, h_choice))
            same = float((logits.argmax(1) == h_choice).float().mean())
        print(
            f"  holder, epoch {epoch}: when {when:.4f} (a chance per step alone: {floor:.4f}); "
            f"where {where:.3f}, the scripted choice the likeliest {same:.0%}",
            flush=True,
        )


def _join(parts: list, key: str) -> np.ndarray:
    found = [p[key] for p in parts if len(p[key])]
    return np.concatenate(found) if found else np.zeros(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("start", type=Path, help="Folder of the minds to start from")
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--point-seconds", type=float, default=70.0)
    parser.add_argument("--held-apart", type=float, default=0.1, help="Share of the points")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--stall-reward", type=float, default=STALL_REWARD, help="As train_ppo.py's"
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/models/movement"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_scripted_copy"
    run.mkdir(parents=True)
    print(f"Written to {run}", flush=True)

    began = time.perf_counter()
    seeds = [int(s) for s in rng.integers(0, 2**31, args.points)]
    jobs = [
        (seeds[w :: args.workers], args.point_seconds, args.stall_reward)
        for w in range(args.workers)
    ]
    with Pool(args.workers) as pool:
        parts = []
        for part in pool.imap_unordered(record, jobs):
            parts.append(part)
            print(
                f"played {len(parts)} of {args.workers} workers' points, "
                f"{time.perf_counter() - began:.0f} s",
                flush=True,
            )
    # Kept apart: the last workers' points
    apart = max(1, int(round(args.held_apart * len(parts))))
    learnt, held = parts[:-apart], parts[-apart:]

    brains = Brains.from_minds(load_minds(args.start))
    for side, attacking in (("offence", True), ("defence", False)):
        policy = getattr(brains, side)

        def of(group, key, mask_key):
            values = _join(group, key)
            mask = _join(group, mask_key).astype(bool) == attacking
            return values[mask].astype(np.float32)

        print(f"{side}: {int(of(learnt, 'side', 'side').size)} steps of running", flush=True)
        learn_running(
            policy,
            of(learnt, "x", "side"),
            of(learnt, "run", "side"),
            of(held, "x", "side"),
            of(held, "run", "side"),
            args.epochs,
            rng,
            side,
        )
        learn_values(
            policy,
            of(learnt, "value_x", "value_side"),
            of(learnt, "returned", "value_side"),
            of(held, "value_x", "value_side"),
            of(held, "returned", "value_side"),
            max(1, args.epochs // 3),
            rng,
            side,
        )
    throwing = {k: _join(learnt, k) for k in ("throw", "threw", "choices", "prior", "choice")}
    held_throwing = {k: _join(held, k) for k in ("throw", "threw", "choices", "prior", "choice")}
    print(f"holder: {len(throwing['throw'])} steps, {len(throwing['choice'])} throws", flush=True)
    learn_throwing(brains.offence, throwing, held_throwing, args.epochs, rng)
    save_brains(brains, run, copied_from="scripted", start=str(args.start))
    print(f"kept in {run}", flush=True)

    # The copy against the scripted team, as it would play without trying things
    scored = conceded = throws = catches = 0
    for k in range(32):
        point_rng = np.random.default_rng(10_000 + k)
        side = k % 2
        ours = PolicyTeam(brains, side, point_rng, record=False, greedy=True)
        theirs = ScriptedTeam(k)
        teams = (ours, theirs) if side == 0 else (theirs, ours)
        start = game.pull_start(point_rng, int(point_rng.integers(2)), game.a_wind(point_rng))
        match = Match(teams, point_rng, start)
        while not match.state.over and match.state.seconds < 120:
            match.step()
        scored += match.state.scored == side
        conceded += match.state.scored == 1 - side
        for event in match.state.events:
            if event.team == side:
                throws += event.kind == "throw"
                catches += event.kind == "catch"
    print(
        f"the copy against the scripted team: scored {scored}, conceded {conceded} of 32 "
        f"points; completed {catches}/{throws}",
        flush=True,
    )


if __name__ == "__main__":
    main()
