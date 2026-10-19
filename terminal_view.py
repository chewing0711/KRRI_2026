"""pi-hw main.py의 result와 sound_frame_half를 표시하는 전용 모듈.

수집/추론/장치 접근은 하지 않는다. 전달받은 0.5초 파형에서 100ms RMS
5개를 계산한다. 그래프는 최근 100개 구간(수집된 오디오 10초)을 표시하고,
오른쪽 값은 마지막 100ms RMS이다. 갱신은 기존 main.py 루프당 한 번이다.
"""
from collections import deque

import numpy as np
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class TerminalView:
    def __init__(self):
        self.console = Console()
        self.history = deque(maxlen=100)
        self.result = None
        self.status = "WAITING"
        self.current_rms = None
        self.live = None

    def __enter__(self):
        self.live = Live(
            self.render(), console=self.console, screen=True,
            auto_refresh=False, vertical_overflow="crop",
        )
        self.live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.live is not None:
            self.live.stop()
        return False

    def update_audio(self, sound_frame_half, sample_rate=48000):
        """마이크 파형만 받아 RMS 그래프와 우하단 값을 갱신한다."""
        audio = np.asarray(sound_frame_half, dtype=np.float64)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("유효한 mono 오디오 파형이 필요합니다.")
        samples = round(sample_rate * 0.1)
        if samples < 1 or audio.size % samples:
            raise ValueError("오디오는 100ms 구간으로 나눌 수 있어야 합니다.")
        windows = audio.reshape(-1, samples)
        rms_values = np.sqrt(np.mean(windows * windows, axis=1))
        self.history.extend(float(v) for v in rms_values)
        self.current_rms = float(rms_values[-1])
        self._refresh()

    def update_result(self, result):
        """멀티모달 모델 결과만 좌하단에 표시한다."""
        if not np.isscalar(result) or result not in (0, 1):
            raise ValueError("모델 결과는 0 또는 1이어야 합니다.")
        self.result = int(result)
        self.status = "RESULT"
        self._refresh()

    def set_no_can(self):
        """CAN 데이터 중단: 이전 모델 결과 제거, RMS/그래프는 유지."""
        if self.status != "NO CAN":
            self.result = None
            self.status = "NO CAN"
            self._refresh()

    def update(self, result, sound_frame_half, sample_rate=48000):
        """기존 update(result, audio) 호출과의 호환용."""
        self.update_audio(sound_frame_half, sample_rate)
        self.update_result(result)

    def _refresh(self):
        if self.live is not None:
            self.live.update(self.render(), refresh=True)

    def _graph(self, columns, rows):
        """0~1 고정 Y축. X축은 수집 구간 순서이며 wall-clock 시간이 아니다."""
        pixels = np.zeros((rows, columns), dtype=np.uint8)
        masks = ((1, 8), (2, 16), (4, 32), (64, 128))
        width, height = columns * 2, rows * 4

        def dot(x, y):
            pixels[y // 4, x // 2] |= masks[y % 4][x % 2]

        previous = None
        for i, value in enumerate(self.history):
            x = round(i * (width - 1) / (self.history.maxlen - 1))
            y = round((1 - np.clip(value, 0.0, 1.0)) * (height - 1))
            if previous is None:
                dot(x, y)
            else:
                px, py = previous
                count = max(abs(x - px), abs(y - py), 1)
                for step in range(count + 1):
                    dot(round(px + (x - px) * step / count),
                        round(py + (y - py) * step / count))
            previous = (x, y)

        text = Text(no_wrap=True, overflow="crop")
        for row in range(rows):
            axis = {0: "1.0", rows // 2: "0.5", rows - 1: "0.0"}
            text.append(f"{axis.get(row, ''):>3} │", style="dim")
            cells = "".join(chr(0x2800 + int(v)) if v else " " for v in pixels[row])
            text.append(cells, style="cyan")
            if row != rows - 1:
                text.append("\n")
        return text

    def render(self):
        width, height = self.console.size
        bottom_height = min(7, max(4, height // 3))
        graph_rows = max(1, height - bottom_height - 2)
        graph_columns = max(1, width - 9)

        result_text = ("NO CAN" if self.status == "NO CAN" else
                       "WAITING" if self.status == "WAITING" else
                       "ABNORMAL" if self.result == 1 else "NORMAL")
        level_text = "--" if self.current_rms is None else f"{self.current_rms:.4f}"
        result_style = ("bold yellow" if self.status == "NO CAN" else
                        "bold" if self.status == "WAITING" else
                        "bold red" if self.result == 1 else "bold green")
        layout = Layout()
        layout.split_column(
            Layout(Panel(self._graph(graph_columns, graph_rows),
                         title="사운드 그래프 · RMS / 100ms",
                         border_style="cyan"), name="sound"),
            Layout(name="values", size=bottom_height),
        )
        layout["values"].split_row(
            Layout(Panel(Align.center(Text(result_text, style=result_style),
                                      vertical="middle"), title="최종 판정")),
            Layout(Panel(Align.center(Text(level_text, style="bold"),
                                      vertical="middle"), title="현재 RMS")),
        )
        return layout
