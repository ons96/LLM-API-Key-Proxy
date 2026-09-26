"""#928 chain-regeneration runner tests (fixture-DB dry-run cycles).

Run: python3 -m pytest tests/test_regenerate_chains.py -q
  or: python3 -m unittest discover -s tests -p 'test_regenerate_chains.py'
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import regenerate_chains as rc  # noqa: E402


def _write(tmp, name, doc):
    p = tmp / name
    p.write_text(yaml.safe_dump(doc))
    return p


def _fixtures(tmp):
    providers_db = {
        "providers": [
            {
                "id": "dummy-free",
                "enabled": True,
                "free_tier": True,
                "capabilities": ["chat"],
                "free_models": [
                    {
                        "id": "dummy-coder-1",
                        "capabilities": ["code"],
                        "context": 131072,
                        "tps": 60,
                    }
                ],
            },
            # must never enter the pool: not free_tier
            {"id": "paid-prov", "enabled": True, "free_tier": False,
             "free_models": [{"id": "paid-model", "capabilities": ["code"]}]},
            # must never enter the pool: disabled
            {"id": "disabled-prov", "enabled": False, "free_tier": True,
             "free_models": [{"id": "disabled-model", "capabilities": ["code"]}]},
        ]
    }
    vm_doc = {
        "metadata": {"generated_by": "fixture"},
        "virtual_models": {
            "coding-fast": {
                "model": "coding-fast",
                "fallback_chain": [
                    {"provider": "groq", "model": "gpt-oss-20b",
                     "reasoning_effort": "None"},
                ],
            },
        },
    }
    empty_rankings = {"models": []}
    return {
        "config": _write(tmp, "virtual_models.yaml", vm_doc),
        "providers_db": _write(tmp, "providers_database.yaml", providers_db),
        "policy": _write(
            tmp,
            "dead_providers.yaml",
            {
                # load_policy requires >=1 direct free fallback; nothing blocked.
                "direct_free_fallbacks": [
                    {"provider": "gemini", "model": "gemini-3-flash-preview",
                     "reasoning_effort": "high"}
                ]
            },
        ),
        "coding_rankings": _write(tmp, "model_rankings.yaml", empty_rankings),
        "chat_rankings": _write(tmp, "chat_model_rankings.yaml", empty_rankings),
    }


class RegenerateChainsTest(unittest.TestCase):
    def _run(self, tmp, apply_changes=False):
        f = _fixtures(tmp)
        before = f["config"].read_text()
        code, report = rc.regenerate(
            config_path=f["config"],
            providers_db_path=f["providers_db"],
            policy_path=f["policy"],
            coding_rankings_path=f["coding_rankings"],
            chat_rankings_path=f["chat_rankings"],
            telemetry_db=str(tmp / "missing-telemetry.db"),
            apply_changes=apply_changes,
        )
        return code, report, f, before

    def test_dummy_provider_enters_pool_in_one_dry_run_cycle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code, report, f, before = self._run(tmp)
            self.assertEqual(code, 0, report.get("log"))
            added = report["added"].get("coding-fast", [])
            self.assertIn("dummy-free/dummy-coder-1", added)
            # paid + disabled providers never enter
            self.assertNotIn("paid-prov/paid-model", added)
            self.assertNotIn("disabled-prov/disabled-model", added)
            # dry-run: nothing written, live config untouched
            self.assertFalse(report["wrote"])
            self.assertEqual(before, f["config"].read_text())

    def test_apply_writes_backed_up_config(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code, report, f, before = self._run(tmp, apply_changes=True)
            self.assertEqual(code, 0)
            self.assertTrue(report["wrote"])
            after = f["config"].read_text()
            self.assertNotEqual(before, after)
            doc = yaml.safe_load(after)
            chain = doc["virtual_models"]["coding-fast"]["fallback_chain"]
            self.assertTrue(
                any(e["provider"] == "dummy-free" for e in chain),
                chain,
            )
            backups = list(f["config"].parent.glob("virtual_models.yaml.bak-pre-regen-*"))
            self.assertEqual(len(backups), 1)

    def test_default_telemetry_db_is_durable_not_devshm(self):
        env = {k: v for k, v in os.environ.items() if k != "TELEMETRY_DB_PATH"}
        probe = (
            "import sys; sys.path.insert(0, 'scripts'); "
            "import reorder_chains as rc; "
            "assert '/dev/shm' not in str(rc.DEFAULT_TELEMETRY_DB), rc.DEFAULT_TELEMETRY_DB; "
            "assert str(rc.DEFAULT_TELEMETRY_DB).endswith('data/telemetry.db')"
        )
        subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT, env=env, check=True
        )


if __name__ == "__main__":
    unittest.main()
