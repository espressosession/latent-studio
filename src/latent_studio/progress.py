"""Live progress for the state panel: two bars and an ETA, fed by patching tqdm
(diffusers' denoise loop and huggingface_hub's downloader are both tqdm
subclasses) rather than a callback threaded through the pipeline."""

import threading
import time
from contextlib import contextmanager

from tqdm.std import tqdm as _std_tqdm

from .icons import svg

_ACCENT = "var(--color-accent, #b4552f)"
_TRACK = "var(--background-fill-secondary, rgba(120,113,108,.18))"


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds in (float("inf"), 0) or seconds < 0:
        return "—"
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h {(total % 3600) // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total}s"


def _amount(value: float, unit: str) -> str:
    if unit != "B":
        return f"{int(value)}"
    for suffix in ("B", "KB", "MB", "GB"):
        if value < 1024 or suffix == "GB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.1f} {suffix}"
        value /= 1024
    return f"{value:.1f} GB"


def _bar(label: str, right: str, fraction: float | None) -> str:
    # fraction None = we know something is running but not how far along.
    width = 100.0 if fraction is None else max(0.0, min(1.0, fraction)) * 100
    opacity = ".35" if fraction is None else "1"
    return (
        '<div style="margin-top:.6rem">'
        '<div style="display:flex;justify-content:space-between;gap:1rem;'
        'font-family:var(--font-mono);font-size:.8rem;opacity:.75;margin-bottom:.3rem">'
        f"<span>{label}</span><span>{right}</span></div>"
        f'<div style="height:7px;border-radius:99px;background:{_TRACK};overflow:hidden">'
        f'<div style="height:100%;width:{width:.1f}%;opacity:{opacity};background:{_ACCENT};'
        'border-radius:99px;transition:width .25s linear"></div></div></div>'
    )


def status_html(icon: str, text: str) -> str:
    """The panel's resting states: ready / done / error."""
    return (
        '<div style="display:flex;align-items:center;gap:.5rem;line-height:1.4">'
        f"{svg(icon)}<span>{text}</span></div>"
    )


def _phase_line(text: str) -> str:
    """The live phase, plain text, no icon — the spinner was the redundant bit (the
    Generate button already shows the coarse state); dropping the text too left the
    panel fully blank during a plain generate, which read as broken rather than idle."""
    return f'<div style="line-height:1.4">{text}</div>'


class ProgressTracker:
    """Shared state between the worker thread (writer) and the UI loop (reader)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "Starting…"
        self.button = "Working…"  # the coarse state, shown on the Generate button
        self._bars: dict[int, dict] = {}  # live tqdm bars, insertion-ordered
        self.image_index = 0
        self.image_total = 1
        self._images_started = time.monotonic()

    # -- written by the worker thread ------------------------------------
    def set_phase(self, phase: str, button: str, image_total: int = 1) -> None:
        with self._lock:
            self.phase = phase
            self.button = button
            self.image_index = 0
            self.image_total = image_total
            self._images_started = time.monotonic()

    def on_image(self, index: int, total: int) -> None:
        """grids.generate_grid() calls this before each cell."""
        with self._lock:
            self.image_index = index
            self.image_total = total

    def bar_open(self, key: int, total, desc: str, unit: str) -> None:
        with self._lock:
            self._bars[key] = {
                "n": 0, "total": total, "desc": desc, "unit": unit, "t0": time.monotonic()
            }

    def bar_update(self, key: int, n: float, total) -> None:
        with self._lock:
            bar = self._bars.get(key)
            if bar is not None:
                bar["n"], bar["total"] = n, total

    def bar_close(self, key: int) -> None:
        with self._lock:
            self._bars.pop(key, None)

    # -- read by the UI loop ---------------------------------------------
    def html(self) -> str:
        """The live phase as plain text (no spinner — the Generate button already
        shows the coarse state, so the icon was the redundant bit, not the words),
        plus whatever progress bars are actually running."""
        with self._lock:
            phase, index, total = self.phase, self.image_index, self.image_total
            started = self._images_started
            # The outermost still-open bar is the meaningful one (e.g. "Fetching N
            # files" rather than one of its per-file children).
            bar = next(iter(self._bars.values()), None)
            bar = dict(bar) if bar else None

        blocks = [_phase_line(phase)]
        inner_fraction = 0.0

        if bar:
            n, bar_total, unit = bar["n"], bar["total"], bar["unit"]
            elapsed = time.monotonic() - bar["t0"]
            if bar_total:
                inner_fraction = n / bar_total
                eta = (bar_total - n) / (n / elapsed) if n and elapsed else None
                right = f"{_amount(n, unit)} / {_amount(bar_total, unit)} · {_duration(eta)} left"
                blocks.append(_bar(bar["desc"] or "Working", right, inner_fraction))
            else:
                blocks.append(_bar(bar["desc"] or "Working", _duration(elapsed), None))

        if total > 1:
            done = index + inner_fraction  # count the image in flight, so the bar creeps
            elapsed = time.monotonic() - started
            eta = (total - done) / (done / elapsed) if done and elapsed else None
            blocks.append(
                _bar(f"Image {min(index + 1, total)} of {total}", f"{_duration(eta)} left", done / total)
            )

        return "".join(blocks)


@contextmanager
def track_tqdm(tracker: ProgressTracker):
    """Routes every tqdm bar created while this is active into `tracker`, by
    patching the shared tqdm base class (global, but the app is single-user)."""
    original = (_std_tqdm.__init__, _std_tqdm.update, _std_tqdm.close)

    def patched_init(self, *args, **kwargs):
        original[0](self, *args, **kwargs)
        if not getattr(self, "disable", False):
            tracker.bar_open(
                id(self), self.total, getattr(self, "desc", "") or "", getattr(self, "unit", "it")
            )

    def patched_update(self, n=1):
        result = original[1](self, n)
        if not getattr(self, "disable", False):
            tracker.bar_update(id(self), self.n, self.total)
        return result

    def patched_close(self):
        tracker.bar_close(id(self))
        return original[2](self)

    _std_tqdm.__init__, _std_tqdm.update, _std_tqdm.close = patched_init, patched_update, patched_close
    try:
        yield tracker
    finally:
        _std_tqdm.__init__, _std_tqdm.update, _std_tqdm.close = original
