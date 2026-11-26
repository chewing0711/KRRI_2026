from collections import deque
import numpy as np
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class TerminalView:
    def __init__(self, max_history=100):
        self.console = Console()
        self.history = deque(maxlen=max_history)
        self.result = None
        self.status = "WAITING"
        self.current_rms = None
        self.live = None

    def __enter__(self):
        self.live = Live(
            self.render(),
            console=self.console,
            screen=True,
            auto_refresh=False,
            vertical_overflow="crop",
        )
        self.live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live is not None:
            self.live.stop()
        return False

    def update_audio(self, audio, sample_rate=48000):
        sig = np.asarray(audio, dtype=np.float64)
        if sig.ndim == 2 and sig.shape[1] == 1:
            sig = sig[:, 0]
        if sig.ndim != 1 or sig.size == 0 or not np.isfinite(sig).all():
            raise ValueError("유효한 mono 오디오 파형이 필요합니다.")
        samples = round(sample_rate * 0.1)
        if samples < 1 or sig.size % samples:
            raise ValueError("오디오는 100ms 구간으로 나눌 수 있어야 합니다.")
        windows = sig.reshape(-1, samples)
        rms_vals = np.sqrt(np.mean(windows * windows, axis=1))
        self.history.extend(float(v) for v in rms_vals)
        self.current_rms = float(rms_vals[-1])
        self.refresh()

    def update_result(self, result):
        if not np.isscalar(result) or result not in (0, 1):
            raise ValueError("모델 결과는 0 또는 1이어야 합니다.")
        self.result = int(result)
        self.status = "RESULT"
        self.refresh()

    def set_no_can(self):
        if self.status != "NO CAN":
            self.result = None
            self.status = "NO CAN"
            self.refresh()

    def update(self, result, audio, sample_rate=48000):
        self.update_audio(audio, sample_rate)
        self.update_result(result)

    def refresh(self):
        if self.live is not None:
            self.live.update(self.render(), refresh=True)

    def draw_graph(self, cols, rows):
        grid = np.zeros((rows, cols), dtype=np.uint8)
        masks = ((1, 8), (2, 16), (4, 32), (64, 128))
        w, h = cols * 2, rows * 4

        def put_dot(x, y):
            grid[y // 4, x // 2] |= masks[y % 4][x % 2]

        prev = None
        for i, val in enumerate(self.history):
            x = round(i * (w - 1) / (self.history.maxlen - 1))
            y = round((1 - np.clip(val, 0.0, 1.0)) * (h - 1))
            if prev is None:
                put_dot(x, y)
            else:
                px, py = prev
                steps = max(abs(x - px), abs(y - py), 1)
                for s in range(steps + 1):
                    put_dot(round(px + (x - px) * s / steps), round(py + (y - py) * s / steps))
            prev = (x, y)

        text = Text(no_wrap=True, overflow="crop")
        for r in range(rows):
            axis = {0: "1.0", rows // 2: "0.5", rows - 1: "0.0"}
            text.append(f"{axis.get(r, ''):>3} │", style="dim")
            cells = "".join(chr(0x2800 + int(v)) if v else " " for v in grid[r])
            text.append(cells, style="cyan")
            if r != rows - 1:
                text.append("\n")
        return text

    def render(self):
        width, height = self.console.size
        bot_h = min(7, max(4, height // 3))
        g_rows = max(1, height - bot_h - 2)
        g_cols = max(1, width - 9)

        label = (
            "NO CAN" if self.status == "NO CAN" else
            "WAITING" if self.status == "WAITING" else
            "ABNORMAL" if self.result == 1 else "NORMAL"
        )
        rms_str = "--" if self.current_rms is None else f"{self.current_rms:.4f}"
        style = (
            "bold yellow" if self.status == "NO CAN" else
            "bold" if self.status == "WAITING" else
            "bold red" if self.result == 1 else "bold green"
        )

        layout = Layout()
        layout.split_column(
            Layout(
                Panel(self.draw_graph(g_cols, g_rows), title="사운드 그래프 · RMS / 100ms", border_style="cyan"),
                name="sound",
            ),
            Layout(name="values", size=bot_h),
        )
        layout["values"].split_row(
            Layout(Panel(Align.center(Text(label, style=style), vertical="middle"), title="최종 판정")),
            Layout(Panel(Align.center(Text(rms_str, style="bold"), vertical="middle"), title="현재 RMS")),
        )
        return layout

    # 호환 별칭
    _refresh = refresh
    _graph = draw_graph
