from SW.rpi_deploy_package.can_inference import inference_can
from SW.rpi_deploy_package.can_parser import collect_can

from sound.sound_main_run import sound_main

from concurrent.futures import ThreadPoolExecutor

from custom_exception import SoundNoDevice, CANNoDevice, check_mic#, check_can

import sounddevice as sd 

import can

def main():
    bus = None
    
    try:
        check_mic()
        # check_can()

        bus = can.Bus(interface="socketcan", channel="can0")
        
        with ThreadPoolExecutor(max_workers=2) as executor:

            while True:
                can_future = executor.submit(collect_can, bus)
                
                sound_future = executor.submit(sound_main)

                can_frame_1s, relative_time = can_future.result()
                result, sound_frame_1s = sound_future.result()

                if result: 
                    print("sound abnormal")
                    result = inference_can(can_frame_1s)
                #    최종판단 = can 동작 결과(can_frame_1s)
                print(f"ble send:{result}")
                # ble_send(최종판단, 1초 사운드 프레임)이거 비동기 멀티쓰레드

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
