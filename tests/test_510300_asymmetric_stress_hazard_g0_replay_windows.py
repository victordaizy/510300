from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
G0_REPLAY = ROOT / "scripts/run_510300_asymmetric_stress_hazard_v1_g0_replay.ps1"


def test_g0_replay_enables_git_long_paths_for_worktree_lifecycle() -> None:
    script = G0_REPLAY.read_text(encoding="utf-8-sig")
    assert '$GitLongPathArgs = @("-c", "core.longpaths=true")' in script
    assert "& git @GitLongPathArgs worktree add --detach" in script
    assert "& git -C $ProjectRoot @GitLongPathArgs worktree remove --force" in script
