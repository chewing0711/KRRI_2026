import warnings
warnings.filterwarnings("ignore")

from SW.rpi_deploy_package.can_inference import inference
from SW.rpi_deploy_package.can_parser import collect_can

from sound.sound_main_run import sound_main

from concurrent.futures import ThreadPoolExecutor

from custom_exception import SoundNoDevice, CANNoDevice, check_mic, check_can

import sounddevice as sd 

import can

# from ble_server import ble_send, load_all_wav_files


def main():
    bus = None

    try:
        check_mic()
        check_can()

        bus = can.Bus(
            interface="socketcan",
            channel="can0"
        )

        with ThreadPoolExecutor(max_workers=2) as executor:

            while True:
                # CAN + Audio 병렬 수집
                can_future = executor.submit(collect_can, bus)
                sound_future = executor.submit(sound_main)

                can_frame_half, relative_time = can_future.result()
                sound_frame_half, audio_features_24d = sound_future.result()

                # 멀티모달 모델 실행
                result = inference(
                    can_frame_half,
                    audio_features_24d
                )

                print(
                    f"Multimodal: {'Abnormal' if result == 1 else 'Normal'}",
                    flush=True
                )

                # BLE 송신
                # ble_send(result, sound_frame_half)

    except SoundNoDevice as e:
        print(e)

    except CANNoDevice as e:
        print(e)

    except KeyboardInterrupt:
        print("Exit.")

    finally:
        if bus is not None:
            bus.shutdown()

if __name__ == "__main__":
    main()

