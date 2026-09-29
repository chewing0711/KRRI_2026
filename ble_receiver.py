# KRRI_2026 무선 수신 및 실시간 상태 모니터링 모듈

import asyncio
import time
import struct
import math
import datetime
import threading
import subprocess
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
from bleak import BleakClient, BleakScanner

# 타겟 송신기 MAC 주소 및 특성 식별자
SERVER_MAC = "2C:CF:67:63:37:AA"
CHARACTERISTIC_UUID = "20260002-cafe-1234-5678-0123456789ab"
AGENT_PATH = "/org/bluez/AutoAgent"
EXPECTED_PACKET_SIZE = 164


class AutoAgent(dbus.service.Object):
    # 블루투스 페어링 요청 시 0초 자동 승인 처리 에이전트
    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Release(self): pass

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="s")
    def RequestPinCode(self, device): return "0000"

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="u")
    def RequestPasskey(self, device): return dbus.UInt32(0)

    @dbus.service.method("org.bluez.Agent1", in_signature="ouq", out_signature="")
    def DisplayPasskey(self, device, passkey, entered): pass

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def DisplayPinCode(self, device, pincode): pass

    @dbus.service.method("org.bluez.Agent1", in_signature="ou", out_signature="")
    def RequestConfirmation(self, device, passkey): return

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="")
    def RequestAuthorization(self, device): return

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def AuthorizeService(self, device, uuid): return

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Cancel(self): pass


def setup_bluez_agent():
    # 백그라운드 스레드에서 BlueZ AgentManager에 자동 승인 에이전트 등록
    def run_agent_loop():
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SystemBus()
        agent = AutoAgent(bus, AGENT_PATH)
        manager = dbus.Interface(bus.get_object("org.bluez", "/org/bluez"), "org.bluez.AgentManager1")
        try:
            manager.RegisterAgent(AGENT_PATH, "NoInputNoOutput")
            manager.RequestDefaultAgent(AGENT_PATH)
            print("자동 승인 등록 완료", flush=True)
        except Exception:
            pass
        GLib.MainLoop().run()

    t = threading.Thread(target=run_agent_loop, daemon=True)
    t.start()


def on_receive(sender, data: bytearray):
    # 무선으로 수신된 바이너리 패킷 파싱 및 실시간 상태 출력
    if len(data) < 2 or data[0] != 0xAA:
        return

    # 시간 및 AI 진단 상태 판정
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_code = data[1]
    status_str = "ABNORMAL" if status_code == 1 else "NORMAL"

    # 20개 서브구간의 실효 음압(RMS, Root Mean Square) 값 추출 및 데시벨(dB) 환산
    num_windows = (len(data) - 2) // 8
    rmse_list = []
    for i in range(num_windows):
        offset = 2 + (i * 8)
        rmse, _ = struct.unpack_from('<ff', data, offset)
        if rmse > 0:
            rmse_list.append(rmse)
    
    avg_rmse = (sum(rmse_list) / len(rmse_list)) if rmse_list else 0.0
    db_val = min(100.0, max(20.0, 20 * math.log10(max(avg_rmse, 1e-4) * 1000) + 40))

    # 표준 규격 포맷으로 출력
    print(f"{current_time} Status: {status_str} ({status_code}) | dB: {db_val:.1f} dB", flush=True)


def clean_bluetooth_environment():
    # 이전 블루투스 연결 상태 및 스캔 초기화
    try:
        subprocess.run(["bluetoothctl", "disconnect", SERVER_MAC], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["bluetoothctl", "scan", "off"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except:
        pass


async def main():
    # BLE 수신 메인 루프 및 연결 감시 데몬
    print(f"BLE 수신기 시작 (타겟 MAC: {SERVER_MAC}, 패킷 규격: {EXPECTED_PACKET_SIZE}B)", flush=True)
    clean_bluetooth_environment()
    setup_bluez_agent()
    await asyncio.sleep(0.5)

    while True:
        try:
            # 1. 타겟 디바이스 탐색
            print(f"송신기({SERVER_MAC}) 검색 중", flush=True)
            device = await BleakScanner.find_device_by_address(SERVER_MAC, timeout=5.0)
            if not device:
                print(f"송신기({SERVER_MAC})를 찾지 못했습니다. 재시도 중", flush=True)
                await asyncio.sleep(0.5)
                continue

            # 2. GATT 서버 연결
            print(f"송신기({SERVER_MAC}) GATT 연결 시도 중", flush=True)
            async with BleakClient(device, timeout=10.0) as client:
                print(f"GATT 연결 성공 (MTU: {client.mtu_size}). 서비스 탐색 중", flush=True)
                
                # 3. 타겟 Characteristic 탐색
                target_char = None
                for s in client.services:
                    for c in s.characteristics:
                        if c.uuid.lower() == CHARACTERISTIC_UUID.lower():
                            target_char = c
                            break
                    if target_char:
                        break

                if not target_char:
                    print("타겟 Characteristic을 찾지 못했습니다. 재검색 중", flush=True)
                    await asyncio.sleep(0.5)
                    continue

                print(f"타겟 Handle({target_char.handle}) 발견. 실시간 모니터링 시작\n", flush=True)
                
                # 4. 실시간 Notification 구독
                await client.start_notify(target_char, on_receive)
                print("실시간 스트리밍 정상 동작 중\n", flush=True)
                
                # 5. 연결 유지 모니터링
                while client.is_connected:
                    await asyncio.sleep(0.5)

                print("연결 해제 감지. 자동 재연결 중", flush=True)

        except Exception as e:
            # 이전 연결 시도 락 및 스캔 상태 강제 초기화
            subprocess.run(["bluetoothctl", "disconnect", SERVER_MAC], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            subprocess.run(["bluetoothctl", "scan", "off"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            print(f"재연결 대기 중: {e}", flush=True)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n수신기 정상 종료")
