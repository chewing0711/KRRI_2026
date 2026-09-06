from SW.rpi_deploy_package.live_can_inference_node import run_live_inference
from sound.sound_util import check_mic


if __name__ == "__main__":

    try:
        # 마이크 사용 가능 여부 체크
        check_mic()


    except Exception:
        print("에러")
        # run_live_inference()