from types import SimpleNamespace

from musubi_tuner.utils.train_utils import get_resume_step


def test_parses_step_from_state_directory_name():
    path = "/workspace/output/qwen2512-ft-step00001800-state"
    assert get_resume_step(path) == 1800


def test_parses_step_from_trailing_slash():
    path = "/workspace/output/qwen2512-ft-step00000300-state/"
    assert get_resume_step(path) == 300


def test_ignores_epoch_state_directory_name():
    path = "/workspace/output/qwen2512-ft-000006-state"
    assert get_resume_step(path) == 0


def test_falls_back_to_scheduler_last_epoch():
    scheduler = SimpleNamespace(last_epoch=1800)
    assert get_resume_step("qwen2512-ft-state", scheduler) == 1800


def test_prefers_directory_name_over_scheduler():
    scheduler = SimpleNamespace(last_epoch=12)
    path = "/workspace/output/qwen2512-ft-step00001800-state"
    assert get_resume_step(path, scheduler) == 1800


def test_returns_zero_without_resume():
    assert get_resume_step(None) == 0
    assert get_resume_step("") == 0
