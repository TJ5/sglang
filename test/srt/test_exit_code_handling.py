import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHONPATH = str(REPO_ROOT / "python")
if PYTHONPATH not in sys.path:
    sys.path.insert(0, PYTHONPATH)

SGLANG_IMPORT_ERROR = None
try:
    import sglang.srt.utils  # noqa: F401
except ModuleNotFoundError as exc:
    SGLANG_IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    SGLANG_IMPORT_ERROR is not None,
    reason=f"SGLang runtime dependencies are unavailable: {SGLANG_IMPORT_ERROR}",
)


def run_python_snippet(source: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        PYTHONPATH
        if not env.get("PYTHONPATH")
        else f"{PYTHONPATH}{os.pathsep}{env['PYTHONPATH']}"
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_kill_current_process_tree_and_exit_returns_nonzero_from_main_thread():
    result = run_python_snippet(
        "from sglang.srt.utils import kill_current_process_tree_and_exit; "
        "print('helper-imported', flush=True); "
        "kill_current_process_tree_and_exit(exit_code=1)"
    )

    assert result.returncode == 1, result.stderr
    assert "helper-imported" in result.stdout
    assert "Traceback" not in result.stderr


def test_kill_current_process_tree_and_exit_returns_nonzero_from_background_thread():
    result = run_python_snippet(
        "import threading; "
        "from sglang.srt.utils import kill_current_process_tree_and_exit; "
        "print('helper-imported', flush=True); "
        "threading.Thread("
        "target=lambda: kill_current_process_tree_and_exit(exit_code=1)"
        ").start(); "
        "threading.Event().wait(5)"
    )

    assert result.returncode == 1, result.stderr
    assert "helper-imported" in result.stdout
    assert "Traceback" not in result.stderr


def test_detokenizer_process_reraises_after_signaling_parent(monkeypatch):
    from sglang.srt.managers import detokenizer_manager

    signals = []

    class FakeParent:
        def send_signal(self, signum):
            signals.append(signum)

    class FakeProcess:
        def parent(self):
            return FakeParent()

    class ExplodingDetokenizerManager:
        def __init__(self, server_args, port_args):
            raise RuntimeError("detokenizer startup failed")

    monkeypatch.setattr(
        detokenizer_manager, "kill_itself_when_parent_died", lambda: None
    )
    monkeypatch.setattr(
        detokenizer_manager, "configure_logger", lambda server_args: None
    )
    monkeypatch.setattr(detokenizer_manager.psutil, "Process", lambda: FakeProcess())
    monkeypatch.setattr(
        detokenizer_manager.setproctitle, "setproctitle", lambda _: None
    )

    with pytest.raises(RuntimeError, match="detokenizer startup failed"):
        detokenizer_manager.run_detokenizer_process(
            SimpleNamespace(tokenizer_worker_num=1),
            SimpleNamespace(),
            detokenizer_manager_class=ExplodingDetokenizerManager,
        )

    assert signals == [signal.SIGQUIT]


@pytest.mark.parametrize(
    "module_name,wrapper_name",
    [
        ("sglang.srt.managers.tokenizer_manager", "print_exception_wrapper"),
        ("sglang.srt.managers.multi_tokenizer_mixin", "print_exception_wrapper"),
    ],
)
def test_async_exception_wrappers_exit_nonzero(monkeypatch, module_name, wrapper_name):
    module = pytest.importorskip(module_name)
    calls = []

    async def raise_error():
        raise RuntimeError("async failure")

    def fake_kill_current_process_tree_and_exit(exit_code, wait_timeout=None):
        calls.append((exit_code, wait_timeout))
        raise SystemExit(exit_code)

    monkeypatch.setattr(
        module,
        "kill_current_process_tree_and_exit",
        fake_kill_current_process_tree_and_exit,
    )

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(getattr(module, wrapper_name)(raise_error))

    assert exc_info.value.code == 1
    assert calls == [(1, None)]
