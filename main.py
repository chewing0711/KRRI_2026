import warnings
warnings.filterwarnings("ignore")

from SW.rpi_deploy_package.can_inference import inference_can
from SW.rpi_deploy_package.can_parser import collect_can

from sound.sound_main_run import sound_main

from concurrent.futures import ThreadPoolExecutor

from custom_exception import SoundNoDevice, CANNoDevice, check_mic, check_can

import sounddevice as sd 

import can

from ble_server import ble_send, load_all_wav_files


def main():
    bus = None
    
    try:
        check_mic()
        check_can()

        bus = can.Bus(interface="socketcan", channel="can0")
        
        with ThreadPoolExecutor(max_workers=2) as executor:

            while True:
                can_future = executor.submit(collect_can, bus)
                
                sound_future = executor.submit(sound_main)

                can_frame_1s, relative_time = can_future.result()
                result, sound_frame_1s = sound_future.result()
                
                # False 정상, True 비정상
                # 사운드 (0: 정상, 1: 고장)
                if result['prediction']: 
                    can_result = inference_can(can_frame_1s)
                    # 0이 정상, 1이 이상
                    print(f"CAN 판단: {'Normal' if (can_result == 0) else 'Abnormal'}")
                    ble_send(can_result, sound_frame_1s)
                    continue
                
                print(f"Sound 판단: {result['prediction']}, 정상?", flush=True)
                ble_send(result['prediction'], sound_frame_1s)

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

