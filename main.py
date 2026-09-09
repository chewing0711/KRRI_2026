from SW.rpi_deploy_package.can_inference import can_main
from sound.sound_main_run import sound_main

from concurrent.futures import ThreadPoolExecutor

from custom_exception import SoundNoDevice, CANNoDevice, check_mic#, check_can

def main():
    try:
        check_mic()
        # check_can()
        
        with ThreadPoolExecutor(max_workers=2) as executor:

            while True:
                # can_future = executor.submit(can_1초 수집함수)
                sound_future = executor.submit(sound_main)

                # can_frame_1s = can_future.result()
                sound_result, sound_frame_1s = sound_future.result()


                # if sound가 abnomarl:
                #    최종판단 = can 동작 결과(can_frame_1s)

                # ble_send(최종판단, 1초 사운드 프레임)이거 비동기 멀티쓰레드

    except SoundNoDevice as e:
        print(e)

    except CANNoDevice as e:
        print(e)

if __name__ == "__main__":
    main()
