"""Teams that learn where to go and where to throw, from nothing (simulation/spots.py)."""

import unittest

import numpy as np
import torch
from support import load_module

game = load_module("simulation.game")
perception = load_module("simulation.perception")
spots = load_module("simulation.spots")
sf = load_module("simulation.spot_features")
scripted = load_module("simulation.scripted")
match = load_module("simulation.match")


def first_views(team=0):
    state = game.line_up(np.random.default_rng(0))  # Team 0 holds the disc
    still = game.Decision(np.zeros((game.PLAYERS, 2)))
    game.step(state, (still, still), np.random.default_rng(0))
    return state, perception.Senses(team, np.random.default_rng(0)).look(state)


class ChoicesTest(unittest.TestCase):
    def test_a_player_may_stay_cut_or_run_beside_anyone_but_themselves(self):
        _, views = first_views()
        features, kinds, players, params, allowed = sf.spot_choices(sf.views_of(views))
        self.assertEqual(tuple(features.shape), (game.PLAYERS, sf.SPOT_CHOICES, sf.SPOT_FEATURES))
        self.assertEqual(tuple(allowed.shape), (game.PLAYERS, sf.SPOT_CHOICES))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(bool(allowed[:, sf.STAY].all()))
        self.assertTrue(bool(allowed[:, sf.FIRST_CUT : sf.FIRST_AROUND].all()))
        self.assertFalse(bool(allowed[:, sf.KEEP].any()))  # Nowhere to go on to yet
        self.assertTrue(bool(allowed[:, sf.FIRST_OPPONENT :].all()))  # Any opponent
        mates = allowed[:, sf.FIRST_TEAMMATE : sf.FIRST_OPPONENT].reshape(7, 7, -1)
        own = torch.eye(7, dtype=torch.bool)
        self.assertFalse(bool(mates[own].any()))  # Not beside themselves
        self.assertTrue(bool(mates[~own].all()))
        # Around the disc
        around = slice(sf.FIRST_AROUND, sf.FIRST_TEAMMATE)
        self.assertTrue(bool(allowed[:, around].any(1).all()))

    def test_a_cut_starts_from_the_player_and_ends_on_the_field(self):
        _, views = first_views()
        view = views[3]
        _, kinds, _, params, _ = sf.spot_choices(sf.views_of([view]))
        me = view.ours[3, :2]
        np.testing.assert_allclose(params[0, sf.STAY].numpy(), me, atol=1e-4)
        cuts = params[0, sf.FIRST_CUT : sf.FIRST_AROUND].numpy()
        self.assertTrue(bool((kinds[0, : sf.FIRST_TEAMMATE] == sf.FIELD).all()))
        self.assertGreaterEqual(float(cuts.min()), sf.EDGE - 1e-4)
        self.assertLessEqual(float(cuts[:, 0].max()), game.WIDTH - sf.EDGE + 1e-4)
        self.assertLessEqual(float(cuts[:, 1].max()), game.LENGTH - sf.EDGE + 1e-4)
        inside = (np.abs(cuts - (me + np.asarray(sf.CUTS))) < 1e-3).all(1)
        self.assertGreater(int(inside.sum()), 0)  # Those not stopped by a line: as drawn
        lengths = np.hypot(*(cuts[inside] - me).T)
        self.assertTrue(set(np.round(lengths).astype(int)) <= set(sf.CUT_LENGTHS))

    def test_a_player_may_go_on_to_where_they_were_going(self):
        _, views = first_views()
        view = views[3]
        plan = (sf.FIELD, -1, np.array([12.0, 55.0]))
        _, _, _, params, allowed = sf.spot_choices(sf.views_of([view], [plan], ages=[0.4]))
        self.assertTrue(bool(allowed[0, sf.KEEP]))
        np.testing.assert_allclose(params[0, sf.KEEP].numpy(), plan[2], atol=1e-4)

    def test_an_attacker_on_their_way_mostly_goes_on_at_first(self):
        _, views = first_views()
        far = (sf.FIELD, -1, views[3].ours[3, :2] + np.array([0.0, 12.0]))
        here = (sf.FIELD, -1, views[3].ours[3, :2] + np.array([0.0, 0.5]))
        for plan, share in ((far, sf.GO_ON_AT_FIRST), (here, None)):
            seeing = sf.views_of([views[3]], [plan], ages=[1.0])
            allowed = sf.spot_choices(seeing)[4]
            chances = torch.softmax(sf.spot_leans(seeing, allowed)[0], 0)
            self.assertEqual(float(chances[~allowed[0]].sum()), 0.0)
            expected = 1.0 / int(allowed.sum()) if share is None else share
            self.assertAlmostEqual(float(chances[sf.KEEP]), expected, places=4)

    def test_the_scores_from_the_parts_are_those_from_the_numbers_written_out(self):
        state, _ = first_views()
        state.pairs = game.match_up(state.places, state.offence)
        torch.manual_seed(0)
        network = spots.SpotBrains().spots["defence"]
        torch.nn.init.normal_(network[-1].weight, std=0.5)
        for team in (0, 1):
            views = perception.Senses(team, np.random.default_rng(0)).look(state)
            plans = [(sf.OPPONENT, 2, sf.BESIDE[0]), None, (sf.FIELD, -1, np.array([9.0, 40.0]))]
            seeing = sf.views_of(views, plans + [None] * 4, ages=[0.3] * 7)
            with torch.no_grad():
                written = network(sf.spot_choices(seeing)[0])[..., 0]
                parts = sf.spot_parts(seeing)
                quick = sf.spot_scores(network, parts)
                some = torch.tensor([True, False, True, False, False, True, False])
                fewer = sf.spot_scores(network, sf.take_parts(parts, some))
            self.assertGreater(float(written.std()), 0.05)
            np.testing.assert_allclose(quick.numpy(), written.numpy(), atol=2e-5)
            np.testing.assert_allclose(fewer.numpy(), written[some].numpy(), atol=2e-5)

    def test_a_defender_may_stand_beside_any_attacker_who_is_followed(self):
        state, _ = first_views()
        state.pairs = game.match_up(state.places, state.offence)
        views = perception.Senses(1, np.random.default_rng(0)).look(state)
        view = views[2]
        guarded = int(view.partner[view.me])
        features, kinds, players, params, allowed = sf.spot_choices(sf.views_of([view]))
        beside = slice(sf.FIRST_OPPONENT, sf.SPOT_CHOICES)
        self.assertTrue(bool(allowed[0, beside].all()))  # A poach is theirs to choose
        self.assertTrue(bool((kinds[0, beside] == sf.OPPONENT).all()))
        first = sf.FIRST_OPPONENT + guarded * len(sf.BESIDE)
        self.assertEqual(int(players[0, first]), guarded)
        # The spots beside the attacker they guard are told apart
        flagged = torch.nonzero(features[0, :, 5 + sf.CHOICE_KINDS - 1])[:, 0].tolist()
        self.assertEqual(flagged, list(range(first, first + len(sf.BESIDE))))
        choice = (sf.OPPONENT, guarded, params[0, first].numpy())
        np.testing.assert_allclose(
            spots._spot_of(view, choice), view.theirs[guarded, :2] + choice[2]
        )
        # ... a moment late: where the attacker was, not where they are
        late = view.theirs[:, :2] - np.array([0.0, 2.0])
        np.testing.assert_allclose(
            spots._spot_of(view, choice, late), view.theirs[guarded, :2] + choice[2] - [0, 2]
        )
        # Going on to that spot is theirs to choose again
        going = sf.spot_choices(sf.views_of([view], [choice], ages=[0.2]))[4]
        self.assertTrue(bool(going[0, sf.KEEP]))

    def test_a_place_with_an_opponent_on_it_is_not_open(self):
        state, views = first_views()
        me = views[3].ours[3, :2]
        cut = sf.FIRST_CUT + 1
        where = sf.spot_choices(sf.views_of([views[3]]))[3][0, cut].numpy()
        state.places[game.PLAYERS + 2] = where
        state.velocities[game.PLAYERS + 2] = 0.0
        view = perception.Senses(0, np.random.default_rng(0)).look(state)[3]
        features = sf.spot_choices(sf.views_of([view]))[0][0].numpy()
        nearest, _, soon, before, _ = features[cut, -5:]
        self.assertAlmostEqual(float(nearest), 0.0, places=4)
        self.assertGreater(float(soon), 0.0)
        self.assertLess(float(before), 0.0)  # They are there already
        self.assertGreater(float(np.hypot(*(where - me))), 1.0)

    def test_a_disc_on_the_ground_is_described_from_each_spot(self):
        state, _ = first_views()
        state.holder, state.disc = None, np.array([20.0, 40.0])
        views = perception.Senses(0, np.random.default_rng(0)).look(state)
        v = sf.views_of(views)
        self.assertTrue(bool((v["disc_mode"] == sf.ON_GROUND).all()))
        features = sf.spot_choices(v)[0]
        on_ground = sf.SPOT_FEATURES - 5 - 7 - 7 - 3 - 3 - 9 + 2  # Third of the disc's
        self.assertTrue(bool((features[..., on_ground] == 1.0).all()))

    def test_every_player_is_described_the_nearest_first(self):
        _, views = first_views()
        v = sf.views_of([views[1]])
        features = sf.spot_choices(v)[0][0].numpy()
        self.assertEqual(sum(count for _, count in sf.SPOT_GROUPS), sf.SPOT_FEATURES)
        self.assertEqual(sum(count for _, count in sf.THROW_GROUPS), sf.THROW_FEATURES)
        start = 5 + sf.CHOICE_KINDS  # After the place and the kind of choice
        mates = features[0, start : start + sf.MATES * sf.PER_PLAYER].reshape(sf.MATES, -1)
        from_me = np.hypot(mates[:, 2], mates[:, 3])
        self.assertTrue(bool((np.diff(from_me) >= -1e-6).all()))  # Nearest first
        self.assertTrue(bool((mates[:, 12] == 1.0).all()))  # All six known
        self.assertEqual(float(mates[:, 10].sum()), 1.0)  # One of them has the disc

    def test_the_holder_may_hold_on_or_throw_anywhere(self):
        _, views = first_views()
        features = sf.throw_choices(sf.views_of([views[0]]))[0]
        self.assertEqual(tuple(features.shape), (sf.THROW_CHOICES, sf.THROW_FEATURES))
        self.assertEqual(float(features[0, 0]), 1.0)  # Holding on
        self.assertEqual(float(features[1:, 0].sum()), 0.0)
        # At first the holder holds on most of the time
        chances = np.exp(spots.THROW_PRIOR) / np.exp(spots.THROW_PRIOR).sum()
        self.assertAlmostEqual(float(chances[0]), spots.HOLD_AT_FIRST, places=5)
        # ... and throws to a teammate half the time it throws
        mates = chances[1 + len(sf.THROW_TARGETS) :].sum() / chances[1:].sum()
        self.assertAlmostEqual(float(mates), sf.MATE_SHARE, places=5)
        self.assertEqual(float(features[1 + len(sf.THROW_TARGETS) :, -1].min()), 1.0)
        self.assertEqual(float(features[: 1 + len(sf.THROW_TARGETS), -1].max()), 0.0)

    def test_a_throw_to_a_teammate_leads_them_as_the_scripted_team_does(self):
        state, _ = first_views()
        state.velocities[3] = (4.0, 6.0)  # A cutter on the move
        views = perception.Senses(0, np.random.default_rng(0)).look(state)
        view = views[state.holder]
        aims = sf.throw_aims(sf.views_of([view]))[0, len(sf.THROW_TARGETS) :].numpy()
        led = scripted.throw_options(view)[:, :2]
        np.testing.assert_allclose(aims, led, atol=1e-3)
        self.assertGreater(
            float(np.abs(aims - view.ours[np.arange(7) != view.holder, :2]).max()), 1.0
        )

    def test_a_team_trained_on_what_players_saw_before_is_refused(self):
        state = spots.SpotBrains().state()
        old = dict(state["spots"]["offence"])
        old["0.weight"] = old["0.weight"][:, :-1]
        state["spots"] = {"offence": old, "defence": dict(old)}
        with self.assertRaisesRegex(ValueError, "train a new team"):
            spots.SpotBrains().load(state)


class SpotTeamTest(unittest.TestCase):
    def test_a_team_that_knows_nothing_plays_a_point_and_keeps_its_choices(self):
        rng = np.random.default_rng(1)
        ours = spots.SpotTeam(spots.SpotBrains(), 0, rng)
        start = game.pull_start(rng, 0)
        point = match.play_point((ours, scripted.ScriptedTeam(2)), rng, start, limit=20.0)
        self.assertGreater(point.seconds, 0.0)
        self.assertGreater(len(ours.choices), game.PLAYERS)
        sides = {c.side for c in ours.choices}
        self.assertTrue(sides <= {"offence", "defence", "throw"})
        for c in ours.choices:
            self.assertEqual(c.value_inputs.shape[0], load_module("simulation.ppo").CRITIC_INPUTS)


if __name__ == "__main__":
    unittest.main()
