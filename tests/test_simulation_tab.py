"""The Simulation tab plays a point with the teams picked, in 3D or from above, and shows
a picked player's brain."""

import unittest

from support import load_module, qt_app


class SimulationTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt_app()
        cls.module = load_module("gui.simulation.simulation_tab")

    def setUp(self):
        self.tab = self.module.SimulationTab()
        self.tab.resize(1200, 800)
        for combo in self.tab.team_combos:
            combo.setCurrentIndex(0)  # Scripted: needs no network

    def test_the_scripted_team_can_always_be_picked(self):
        self.assertEqual(self.module.teams_available()[0], ("Scripted", "scripted"))

    def test_a_point_is_played_step_by_step_in_3d(self):
        self.tab.new_point()
        for _ in range(30):
            self.tab.step()
        self.assertGreaterEqual(self.tab.match.state.seconds, 2.9)
        self.assertEqual(len(self.tab.view3d.scene.players), 14)
        self.assertGreater(self.tab.events.count(), 0)

    def test_from_above_the_field_is_drawn(self):
        self.tab.from_above.setChecked(True)
        self.tab.new_point()
        self.tab.step()
        self.assertFalse(self.tab.flat.pixmap().isNull())

    def test_a_picked_scripted_player_shows_what_they_know_and_no_network(self):
        self.tab.new_point()
        self.tab.step()
        self.tab._pick(9)
        self.assertIn("Red 3", self.tab.brain.title.text())
        self.assertFalse(self.tab.brain.knowledge.pixmap().isNull())
        self.assertIn("scripted", self.tab.brain.network.message)


if __name__ == "__main__":
    unittest.main()
