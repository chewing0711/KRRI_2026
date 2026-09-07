from SW.rpi_deploy_package.can_inference import can_main
from sound.sound_util import check_mic, check_sound
from sound.sound_main_run import sound_main

if __name__ == "__main__":
    try:
        # 마이크 사용 가능 여부 체크
        check_mic()
        if not(check_sound()):
            print("소리의 크기가 너무 작음")
            raise RuntimeError("마이크가 연결되어 있지 않습니다.")
            
        sound_main()

    except Exception:
        can_main()