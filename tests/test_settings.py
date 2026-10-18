"""Every setting the code reads is in configs/default.yaml, with the same default."""

import ast
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MISSING = object()


def settings_read():
    """(file:line, key, default or MISSING) for every get_setting call with a literal key."""
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name != "get_setting" or not node.args:
                continue
            if not isinstance(node.args[0], ast.Constant) or not isinstance(
                node.args[0].value, str
            ):
                continue  # A key put together at run time
            default = MISSING
            if len(node.args) > 1:
                try:
                    default = ast.literal_eval(node.args[1])
                except ValueError:
                    continue  # A default worked out at run time
            yield f"{path.relative_to(ROOT)}:{node.lineno}", node.args[0].value, default


def configured(config: dict, key: str):
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text("utf-8"))

    def test_every_key_read_is_in_the_defaults(self):
        missing = [
            f"{where}: {key}"
            for where, key, _ in settings_read()
            if configured(self.config, key) is MISSING
        ]
        self.assertEqual(missing, [], "settings read but not in configs/default.yaml")

    def test_the_code_falls_back_to_what_the_defaults_say(self):
        differ = []
        for where, key, default in settings_read():
            value = configured(self.config, key)
            if default is MISSING or value is MISSING:
                continue
            same = default == value or (
                isinstance(default, (int, float))
                and isinstance(value, (int, float))
                and not isinstance(default, bool)
                and float(default) == float(value)
            )
            if not same:
                differ.append(f"{where}: {key} = {default!r} in the code, {value!r} in the file")
        self.assertEqual(differ, [], "defaults in the code that differ from default.yaml")


if __name__ == "__main__":
    unittest.main()
