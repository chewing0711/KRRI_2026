"""pi-hw: CAN + Audio 멀티모달 추론과 터미널 모니터.

기존 collect_can / sound_main / inference 재사용.
CAN 데이터가 2초 동안 없으면 NO CAN으로 표시하고 오디오 RMS는 계속 갱신.
"""
import warnings
warnings.filterwarnings("ignore")

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from SW.rpi_deploy_package.can_inference import inference
from SW.rpi_deploy_package.can_parser import collect_can
from sound.sound_main_run import sound_main
from custom_exception import SoundNoDevice, CANNoDevice, check_mic, check_can
from terminal_view import TerminalView
import can

import subprocess
import sys
from pathlib import Path

# BLE 필요시 기존 코드 그대로 주석 해제
from ble_server import ble_send

BASE_DIR = Path(__file__).resolve().parent

CAN_TIMEOUT_SECONDS = 2.0
MAX_AUDIO_CAN_GAP_SECONDS = 0.75


def main():
    bus = None
    ble_process = None
    ble_log = None

    try:
        check_mic()
        check_can()

        bus = can.Bus(interface="socketcan", channel="can0")

        # BLE 서버 자동 실행
        # Rich 터미널 화면과 로그가 섞이지 않도록 파일에 기록
        ble_log = open(
            BASE_DIR / "ble_server.log",
            "a",
            buffering=1,
            encoding="utf-8"
        )

        ble_process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(BASE_DIR / "ble_server.py")
            ],
            cwd=BASE_DIR,
            stdin=subprocess.DEVNULL,
            stdout=ble_log,
            stderr=subprocess.STDOUT
        )

        # BLE 서버 시작 대기
        time.sleep(2)

        if ble_process.poll() is not None:
            raise RuntimeError(
                "BLE 서버 실행 실패. ble_server.log 확인"
            )

        with ThreadPoolExecutor(max_workers=2) as executor, TerminalView() as view:
            # CAN과 마이크를 각각 독립적으로 수집한다.
            can_future = executor.submit(collect_can, bus, CAN_TIMEOUT_SECONDS)
            sound_future = executor.submit(sound_main)
            latest_audio = None
            pending_can = None

            while True:
                if ble_process.poll() is not None:
                    raise RuntimeError(
                        "BLE 서버가 종료되었습니다. ble_server.log 확인"
                )
                done, _ = wait(
                    [can_future, sound_future],
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue

                # 사운드는 CAN 유무와 상관없이 0.5초마다 계속 갱신한다.
                if sound_future in done:
                    sound_frame_half, audio_features_24d = sound_future.result()
                    finished_at = time.monotonic()
                    sound_future = executor.submit(sound_main)
                    view.update_audio(sound_frame_half)
                    latest_audio = (sound_frame_half, audio_features_24d, finished_at)

                if can_future in done:
                    can_frame_half, relative_time = can_future.result()
                    finished_at = time.monotonic()
                    can_future = executor.submit(collect_can, bus, CAN_TIMEOUT_SECONDS)
                    if can_frame_half is None:
                        # 2초간 유효한 CAN 데이터 없음 → 이전 결과 무효화
                        pending_can = None
                        view.set_no_can()
                    else:
                        pending_can = (can_frame_half, finished_at)

                # CAN과 Audio 모두 최근 데이터를 확보했을 때만 멀티모달 추론.
                if pending_can is not None and latest_audio is not None:
                    can_frame_half, can_at = pending_can
                    sound_frame_half, audio_features_24d, audio_at = latest_audio
                    if abs(can_at - audio_at) <= MAX_AUDIO_CAN_GAP_SECONDS:
                        result = int(inference(can_frame_half, audio_features_24d))
                        view.update_result(result)
                        ble_send(result, sound_frame_half)
                        pending_can = None
                        latest_audio = None
                    elif can_at < audio_at:
                        # 오디오보다 오래된 CAN 결과는 버림.
                        pending_can = None
                    else:
                        # CAN보다 오래된 오디오 특징은 버림.
                        latest_audio = None

    except (SoundNoDevice, CANNoDevice) as e:
        print(e)
    except KeyboardInterrupt:
        print("Exit.")
    finally:
        if bus is not None:
            bus.shutdown()

        # main.py 종료 시 BLE 서버도 종료
        if ble_process is not None:
            if ble_process.poll() is None:
                ble_process.terminate()

                try:
                    ble_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ble_process.kill()
                    ble_process.wait()

        if ble_log is not None:
            ble_log.close()


if __name__ == "__main__":
    main()
