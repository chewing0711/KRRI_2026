#!/usr/bin/env python3
"""KRRI_2026: BLE 164-byte receiver with main.py-style Rich monitoring.

Sender layout (little-endian):
    uint8 magic(0xAA), uint8 prediction(0/1),
    5 x [float32 total_rms + 7 x float32 FFT-band rms], uint16 sequence.

The receiver is intentionally independent of dbus-python/PyGObject: Bleak uses
native OS BLE backends. Use on the *receiver* computer, not the RPi transmitter.

Install: python -m pip install bleak 'rich>=13'
Run:     python ble_receiver.py
          python ble_receiver.py --csv received_rms.csv
          python ble_receiver.py --address AA:BB:CC:DD:EE:FF
          python ble_receiver.py --demo     # monitoring UI without hardware
          python ble_receiver.py --auto-scale  # show small RMS variations
          python ble_receiver.py --plain    # text logs, no animated dashboard
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

# Must match pi-sw/ble_server.py
TARGET_NAME = "easylab"
SERVICE_UUID = "20260001-cafe-1234-5678-0123456789ab"
CHARACTERISTIC_UUID = "20260002-cafe-1234-5678-0123456789ab"
MAGIC = 0xAA
PACKET_SIZE = 164
WINDOW_COUNT = 5
FLOATS_PER_WINDOW = 8
ATT_MTU_REQUIRED = PACKET_SIZE + 3
BANDS = (
    "0–100 Hz",
    "100–500 Hz",
    "500–1000 Hz",
    "1–2 kHz",
    "2–5 kHz",
    "5–10 kHz",
    "10–24 kHz",
)
BAND_CSV_FIELDS = (
    "rms_0_100_hz",
    "rms_100_500_hz",
    "rms_500_1000_hz",
    "rms_1000_2000_hz",
    "rms_2000_5000_hz",
    "rms_5000_10000_hz",
    "rms_10000_24000_hz",
)
PACKET_FORMAT = struct.Struct("<BB40fH")
assert PACKET_FORMAT.size == PACKET_SIZE


@dataclass(frozen=True)
class Packet:
    prediction: int
    sequence: int
    windows: tuple[tuple[float, ...], ...]
    overall_rms: float
    band_rms: tuple[float, ...]


def decode_packet(data: bytes | bytearray | memoryview) -> Packet:
    """Decode an entire BLE notification. Reject malformed/partial packets."""
    if len(data) != PACKET_SIZE:
        raise ValueError(f"패킷 길이 오류: {len(data)}B (기대 {PACKET_SIZE}B; ATT MTU >= {ATT_MTU_REQUIRED} 필요)")

    magic, prediction, *fields = PACKET_FORMAT.unpack(data)
    if magic != MAGIC:
        raise ValueError(f"헤더 오류: 0x{magic:02X}")
    if prediction not in (0, 1):
        raise ValueError(f"판정 코드 오류: {prediction}")

    floats = fields[:-1]
    sequence = int(fields[-1])
    if not all(math.isfinite(x) and -1e-7 <= x <= 1.0001 for x in floats):
        raise ValueError("RMS 데이터 범위 오류 (0~1) 또는 NaN/Inf")

    windows = tuple(
        tuple(float(x) for x in floats[i * FLOATS_PER_WINDOW:(i + 1) * FLOATS_PER_WINDOW])
        for i in range(WINDOW_COUNT)
    )
    # Energy-consistent combination of five 100ms RMS values (not arithmetic mean).
    overall_rms = math.sqrt(sum(w[0] ** 2 for w in windows) / WINDOW_COUNT)
    band_rms = tuple(
        math.sqrt(sum(w[j + 1] ** 2 for w in windows) / WINDOW_COUNT)
        for j in range(len(BANDS))
    )
    return Packet(prediction, sequence, windows, overall_rms, band_rms)


@dataclass
class MonitorState:
    connection: str = "검색 대기"
    device: str = "—"
    mtu: int | None = None
    last_error: str = ""
    packet: Packet | None = None
    packets_received: int = 0
    packets_lost: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    invalid_packets: int = 0
    last_sequence: int | None = None
    last_received_monotonic: float | None = None
    rms_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    receive_times: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    on_packet: Callable[[Packet, str], None] | None = None

    def new_connection(self, device: str, mtu: int | None) -> None:
        self.device = device
        self.mtu = mtu
        self.last_error = ""
        self.last_sequence = None  # session reset; sequence might restart after RPi reboot
        self.packet = None  # do not display old session's readings as current data
        self.last_received_monotonic = None
        self.receive_times.clear()
        self.rms_history.clear()
        self.connection = "Notification 구독 중"

    def ingest(self, payload: bytes | bytearray | memoryview) -> None:
        try:
            packet = decode_packet(payload)
        except (TypeError, ValueError, struct.error) as exc:
            self.invalid_packets += 1
            self.last_error = str(exc)
            return

        if self.last_sequence is not None:
            gap = (packet.sequence - self.last_sequence) & 0xFFFF
            if gap == 0:
                self.duplicates += 1
                return
            if gap > 32768:
                self.out_of_order += 1
                return
            if gap > 1:
                self.packets_lost += gap - 1

        self.last_sequence = packet.sequence
        self.packet = packet
        now = time.monotonic()
        self.last_received_monotonic = now
        self.receive_times.append(now)
        self.packets_received += 1
        self.rms_history.extend(w[0] for w in packet.windows)
        self.last_error = ""  # clear stale packet errors after a valid packet

        if self.on_packet is not None:
            self.on_packet(packet, datetime.now().astimezone().isoformat(timespec="milliseconds"))

    def age_seconds(self) -> float | None:
        if self.last_received_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self.last_received_monotonic)

    def packets_per_second(self) -> float:
        if len(self.receive_times) < 2:
            return 0.0
        span = self.receive_times[-1] - self.receive_times[0]
        return (len(self.receive_times) - 1) / span if span > 0 else 0.0


def sparkline(samples: deque[float], width: int = 65) -> tuple[str, float]:
    if not samples:
        return "데이터 대기 중", 0.0
    vals = list(samples)
    if len(vals) > width:
        # Keep transients when downsampling the last 100x100ms windows.
        vals = [max(vals[int(i * len(vals) / width):int((i + 1) * len(vals) / width)] or [0.0])
                for i in range(width)]
    ceiling = max(0.001, max(vals) * 1.1)
    symbols = "▁▂▃▄▅▆▇█"
    return "".join(symbols[min(7, max(0, int(v / ceiling * 7)))] for v in vals), ceiling


def graph_rms(history: deque[float], columns: int, rows: int, auto_scale: bool = False):
    """Use the same Braille RMS plot and fixed 0..1 axis as main/terminal_view.py.

    The source contains 100 ms RMS samples already decoded from BLE packets.
    Auto scaling is opt-in so that the default graph matches main.py.
    """
    from rich.text import Text

    columns = max(1, int(columns))
    rows = max(1, int(rows))
    dots = [[0] * columns for _ in range(rows)]
    masks = ((1, 8), (2, 16), (4, 32), (64, 128))
    width, height = columns * 2, rows * 4
    samples = list(history)
    y_max = max(0.001, max(samples, default=0.0) * 1.10) if auto_scale else 1.0

    def point(x: int, y: int) -> None:
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        dots[y // 4][x // 2] |= masks[y % 4][x % 2]

    previous = None
    for i, value in enumerate(samples[-100:]):
        # Same left-anchored, 100-window scale as the Pi's terminal_view.py.
        x = round(i * (width - 1) / 99)
        y = round((1.0 - min(max(value, 0.0) / y_max, 1.0)) * (height - 1))
        if previous is None:
            point(x, y)
        else:
            px, py = previous
            steps = max(abs(x - px), abs(y - py), 1)
            for step in range(steps + 1):
                point(round(px + (x - px) * step / steps),
                      round(py + (y - py) * step / steps))
        previous = (x, y)

    graph = Text(no_wrap=True, overflow="crop")
    for row in range(rows):
        if row == 0:
            axis_label = f"{y_max:.3f}" if auto_scale else "1.0"
        elif row == rows // 2:
            axis_label = f"{0.5 * y_max:.3f}" if auto_scale else "0.5"
        elif row == rows - 1:
            axis_label = "0.0"
        else:
            axis_label = ""
        graph.append(f"{axis_label:>6} │", style="dim")
        graph.append(
            "".join(chr(0x2800 + value) if value else " " for value in dots[row]),
            style="cyan"
        )
        if row < rows - 1:
            graph.append("\n")
    return graph


def render_dashboard(state: MonitorState, width: int, height: int = 24,
                     detailed: bool = False, auto_scale: bool = False):
    """Render the same large graph + two bottom panels as main.py.

    A compact BLE footer shows connection, sequence, and packet-loss metadata.
    The optional details panel displays the seven combined FFT-band RMS values.
    """
    from rich.align import Align
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    age = state.age_seconds()
    fresh = (state.connection == "수신 중" and state.packet is not None and
             age is not None and age <= 3.0)
    if fresh:
        result_text = "ABNORMAL" if state.packet.prediction else "NORMAL"
        result_style = "bold red" if state.packet.prediction else "bold green"
    elif state.connection == "수신 중" and state.packet is not None:
        result_text, result_style = "NO DATA", "bold yellow"
    elif state.connection in {"연결 해제 / 재검색", "오류 / 재연결 대기"}:
        result_text, result_style = "DISCONNECTED", "bold yellow"
    else:
        result_text, result_style = "WAITING", "bold yellow"

    # Exactly like pi-hw TerminalView: most recent 100 ms, not the 0.5s
    # energy-combined RMS. On loss/stall show '--' rather than stale RMS.
    current_rms = (f"{state.packet.windows[-1][0]:.4f}"
                   if fresh and state.packet is not None else "--")

    bottom_height = min(7, max(4, height // 3))
    footer_height = 2
    # Show optional FFT table only if it will fit without squashing graph.
    show_details = detailed and height >= 30
    details_height = 10 if show_details else 0
    graph_rows = max(1, height - bottom_height - footer_height - details_height - 2)
    graph_columns = max(1, width - 11)
    graph = graph_rms(state.rms_history, graph_columns, graph_rows, auto_scale)

    main_layout = Layout()
    areas = [
        Layout(Panel(graph, title="사운드 그래프 · RMS / 100ms",
                     border_style="cyan"), name="sound", ratio=1),
    ]
    if show_details:
        table = Table(expand=True, box=None, show_header=True)
        table.add_column("FFT 대역")
        table.add_column("최근 0.5초 RMS", justify="right")
        table.add_column("상대 크기", ratio=2)
        band_values = state.packet.band_rms if fresh else (0.0,) * len(BANDS)
        max_band = max(1e-8, max(band_values))
        for band_name, band in zip(BANDS, band_values):
            bar = "█" * min(24, round(24 * band / max_band)) if fresh else ""
            table.add_row(band_name, f"{band:.6f}" if fresh else "--", bar)
        areas.append(Layout(Panel(table, title="FFT 7대역 RMS · 최근 패킷",
                                  border_style="magenta"), name="fft", size=details_height))
    areas.extend((Layout(name="values", size=bottom_height),
                  Layout(name="footer", size=footer_height)))
    main_layout.split_column(*areas)
    main_layout["values"].split_row(
        Layout(Panel(Align.center(Text(result_text, style=result_style),
                                  vertical="middle"), title="최종 판정")),
        Layout(Panel(Align.center(Text(current_rms, style="bold"),
                                  vertical="middle"), title="현재 RMS")),
    )

    sequence = state.last_sequence if state.last_sequence is not None else "--"
    mtu = state.mtu if state.mtu is not None else "--"
    status_line = (f"BLE: {state.connection} | Seq: {sequence} | "
                   f"수신: {state.packets_received} | 누락: {state.packets_lost} | "
                   f"속도: {state.packets_per_second():.1f}/s | MTU 표시: {mtu}")
    footer = Text(no_wrap=True, overflow="ellipsis")
    footer.append(status_line, style="dim cyan" if fresh else "yellow")
    footer.append("\n")
    if state.last_error:
        footer.append("오류: " + state.last_error, style="yellow")
    elif detailed and not show_details:
        footer.append("FFT 상세 화면은 터미널 높이 30줄 이상에서 표시됩니다.", style="dim")
    elif state.mtu is not None and state.mtu < ATT_MTU_REQUIRED:
        footer.append("참고: Linux Bleak MTU 표시 23은 실제 협상값과 다를 수 있습니다.", style="dim")
    else:
        footer.append("최근 100개 × 100ms RMS | -1~1 음원 기준 RMS(음압 dB 아님) | Ctrl+C 종료", style="dim")
    main_layout["footer"].update(footer)
    return main_layout


def match_advertisement(device, adv) -> bool:
    """Match *our* peripheral, not arbitrary devices named 'sender'."""
    names = ((getattr(adv, "local_name", None) or ""), (getattr(device, "name", None) or ""))
    if any(TARGET_NAME in name.casefold() or "sender_eas" in name.casefold() for name in names):
        return True
    advertised = getattr(adv, "service_uuids", None) or []
    return SERVICE_UUID.lower() in (uuid.lower() for uuid in advertised)


async def ble_receive_loop(state: MonitorState, address: str | None, scan_timeout: float):
    from bleak import BleakClient, BleakScanner

    while True:
        state.connection = "검색 중"
        state.mtu = None
        try:
            if address:
                device = await BleakScanner.find_device_by_address(address, timeout=scan_timeout)
            else:
                device = await BleakScanner.find_device_by_filter(match_advertisement, timeout=scan_timeout)
            if device is None:
                state.last_error = f"easylab을 {scan_timeout:g}초 동안 찾지 못함"
                await asyncio.sleep(1.0)
                continue

            state.connection = "연결 중"
            state.device = f"{device.name or TARGET_NAME} ({device.address})"
            disconnected = asyncio.Event()
            loop = asyncio.get_running_loop()

            def on_disconnect(_client):
                loop.call_soon_threadsafe(disconnected.set)

            async with BleakClient(device, disconnected_callback=on_disconnect, timeout=15.0) as client:
                state.new_connection(state.device, client.mtu_size)
                target = client.services.get_characteristic(CHARACTERISTIC_UUID)
                if target is None:
                    raise RuntimeError(f"GATT Characteristic 미발견: {CHARACTERISTIC_UUID}")
                if "notify" not in target.properties:
                    raise RuntimeError("해당 Characteristic은 notify를 지원하지 않음")

                def on_notification(_sender, payload: bytearray):
                    state.ingest(payload)

                await client.start_notify(target, on_notification)
                state.connection = "수신 중"
                try:
                    while client.is_connected and not disconnected.is_set():
                        # Also let UI flag a connected-but-silent link (> 3 sec).
                        try:
                            await asyncio.wait_for(disconnected.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    if client.is_connected:
                        try:
                            await client.stop_notify(target)
                        except Exception:
                            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.connection = "오류 / 재연결 대기"
            state.last_error = f"{type(exc).__name__}: {exc}"
        else:
            state.connection = "연결 해제 / 재검색"
        await asyncio.sleep(1.5)


async def demo_loop(state: MonitorState) -> None:
    """Explicit offline mode for testing display/decoder, never actual BLE."""
    state.connection = "수신 중"
    state.device = "데모 (가상 송신기; BLE 연결 아님)"
    state.mtu = 167
    sequence = 0
    while True:
        sequence = (sequence + 1) & 0xFFFF
        # plausible sample RMS values, not actual microphone measurements
        base = 0.012 + 0.005 * (1 + math.sin(sequence * 0.35))
        values = []
        for w in range(5):
            r = base * (0.85 + w * 0.06)
            values.extend([r, *(r * (0.05 + j * 0.055) for j in range(7))])
        simulated = PACKET_FORMAT.pack(MAGIC, int(sequence // 15 % 2), *values, sequence)
        state.ingest(simulated)
        await asyncio.sleep(0.5)


def configure_csv(state: MonitorState, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    if path.stat().st_size == 0:
        writer.writerow(("received_at", "sequence", "status", "window_start_ms", "total_rms", *BAND_CSV_FIELDS))
        f.flush()

    def write_packet(packet: Packet, timestamp: str) -> None:
        for i, window in enumerate(packet.windows):
            writer.writerow((timestamp, packet.sequence, packet.prediction, 100 * i, *window))
        f.flush()

    state.on_packet = write_packet
    return f


def format_plain(packet: Packet, timestamp: str) -> str:
    return (f"[{timestamp}] seq={packet.sequence:05d} "
            f"status={'ABNORMAL' if packet.prediction else 'NORMAL'} "
            f"rms_0.5s={packet.overall_rms:.6f} "
            f"bands_7={[round(x, 6) for x in packet.band_rms]}")


async def run(args) -> None:
    state = MonitorState()
    f = configure_csv(state, args.csv) if args.csv is not None else None
    callback = state.on_packet
    if args.plain:
        def on_packet(packet: Packet, timestamp: str):
            if callback:
                callback(packet, timestamp)
            print(format_plain(packet, timestamp), flush=True)
        state.on_packet = on_packet

    worker = asyncio.create_task(demo_loop(state) if args.demo
                                 else ble_receive_loop(state, args.address, args.scan_timeout))
    try:
        if args.plain:
            print("BLE 수신 시작" if not args.demo else "데모 모드 시작 (실제 BLE 연결 없음)", flush=True)
            while True:
                await asyncio.sleep(1.0)
                if state.last_error:
                    print(f"상태={state.connection}: {state.last_error}", file=sys.stderr, flush=True)
                    state.last_error = ""
        else:
            from rich.console import Console
            from rich.live import Live
            console = Console()
            with Live(render_dashboard(state, console.width, console.height, args.details, args.auto_scale), console=console,
                      screen=not args.no_alt_screen, auto_refresh=False,
                      vertical_overflow="crop") as live:
                while True:
                    await asyncio.sleep(0.25)
                    live.update(render_dashboard(state, console.width, console.height, args.details, args.auto_scale), refresh=True)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        if f is not None:
            f.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="KRRI_2026 164B BLE real-time receiver/monitor")
    parser.add_argument("--address", help="선택: 송신기 MAC 주소. 생략 시 easylab 이름/UUID로 탐색")
    parser.add_argument("--scan-timeout", type=float, default=5.0, help="1회 탐색 제한 시간(초)")
    parser.add_argument("--csv", type=Path, help="선택: 각 100ms 구간의 원본 RMS 및 7대역 RMS를 CSV 기록")
    parser.add_argument("--details", action="store_true", help="높이 30줄 이상에서 FFT 7대역 RMS를 추가 표시")
    parser.add_argument("--auto-scale", action="store_true", help="작은 RMS도 보이도록 그래프 Y축 자동 확대 (기본은 main.py와 같은 0~1 고정축)")
    parser.add_argument("--demo", action="store_true", help="BLE 연결 없이 가상 패킷으로 화면 테스트")
    parser.add_argument("--plain", action="store_true", help="Rich UI 대신 패킷을 한 줄씩 출력")
    parser.add_argument("--no-alt-screen", action="store_true", help="터미널 대체 화면을 사용하지 않음")
    args = parser.parse_args(argv)
    if args.scan_timeout <= 0:
        parser.error("--scan-timeout은 0보다 커야 합니다")

    if not args.demo:
        try:
            import bleak  # noqa: F401
        except ImportError:
            parser.error("Bleak가 필요합니다: python -m pip install bleak")
    if not args.plain:
        try:
            import rich  # noqa: F401
        except ImportError:
            parser.error("Rich가 필요합니다: python -m pip install 'rich>=13'")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nBLE 수신 모니터 종료")


if __name__ == "__main__":
    main()
