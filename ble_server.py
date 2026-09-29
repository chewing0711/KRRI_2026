# KRRI_2026 무선 송신 서버 및 디지털 신호 처리(DSP, Digital Signal Processing) 연산 모듈

import os
import sys
import time
import struct
import numpy as np
import subprocess
from multiprocessing import shared_memory

# Conda 환경 D-Bus 소켓 경로 설정
if "DBUS_SYSTEM_BUS_ADDRESS" not in os.environ:
    os.environ["DBUS_SYSTEM_BUS_ADDRESS"] = "unix:path=/var/run/dbus/system_bus_socket"

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

# D-Bus 및 BlueZ 인터페이스 식별자
BLUEZ_SERVICE_NAME = 'org.bluez'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'
DBUS_PROP_IFACE = 'org.freedesktop.DBus.Properties'
GATT_SERVICE_IFACE = 'org.bluez.GattService1'
GATT_CHRC_IFACE = 'org.bluez.GattCharacteristic1'

SERVICE_UUID = '20260001-cafe-1234-5678-0123456789ab'
CHAR_UUID = '20260002-cafe-1234-5678-0123456789ab'
PACKET_SIZE = 164

global_shm = None
global_client_shm = None
global_seq = 0


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.freedesktop.DBus.Error.InvalidArgs'


class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotSupported'


class Characteristic(dbus.service.Object):
    # BlueZ GATT Characteristic 객체
    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.path + '/char' + str(index)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.descriptors = []
        self.value = bytearray(PACKET_SIZE)
        self.notifying = False
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_CHRC_IFACE: {
                'Service': self.service.get_path(),
                'UUID': self.uuid,
                'Flags': dbus.Array(self.flags, signature='s'),
                'Descriptors': dbus.Array([], signature='o'),
                'Value': dbus.ByteArray(self.value),
                'Notifying': dbus.Boolean(self.notifying),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface == GATT_CHRC_IFACE:
            return self.get_properties()[GATT_CHRC_IFACE]
        raise InvalidArgsException()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        props = self.GetAll(interface)
        if prop in props:
            return props[prop]
        raise InvalidArgsException()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='ssv', out_signature='')
    def Set(self, interface, prop, value):
        raise NotSupportedException()

    @dbus.service.signal(DBUS_PROP_IFACE, signature='sa{sv}as')
    def PropertiesChanged(self, interface, changed, invalidated):
        pass

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='a{sv}', out_signature='ay')
    def ReadValue(self, options):
        offset = int(options.get('offset', 0))
        global global_shm
        try:
            if global_shm is not None:
                val = bytes(global_shm.buf[:PACKET_SIZE])
                if len(val) == PACKET_SIZE and val[0] == 0xAA:
                    return dbus.ByteArray(val[offset:])
        except:
            pass
        return dbus.ByteArray(self.value[offset:])

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='', out_signature='')
    def StartNotify(self):
        if self.notifying:
            return
        self.notifying = True
        print("수신기가 실시간 스트리밍을 구독했습니다", flush=True)

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='', out_signature='')
    def StopNotify(self):
        if not self.notifying:
            return
        self.notifying = False
        print("수신기가 실시간 스트리밍을 중단했습니다", flush=True)


class Service(dbus.service.Object):
    # BlueZ GATT Service 객체
    def __init__(self, bus, index, uuid, primary):
        self.path = '/org/bluez/example/service' + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                'UUID': self.uuid,
                'Primary': dbus.Boolean(self.primary),
                'Includes': dbus.Array([], signature='o'),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface == GATT_SERVICE_IFACE:
            return self.get_properties()[GATT_SERVICE_IFACE]
        raise InvalidArgsException()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        props = self.GetAll(interface)
        if prop in props:
            return props[prop]
        raise InvalidArgsException()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature='ssv', out_signature='')
    def Set(self, interface, prop, value):
        raise NotSupportedException()


class Application(dbus.service.Object):
    # BlueZ GATT Application 루트 객체
    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            for chrc in service.characteristics:
                response[chrc.get_path()] = chrc.get_properties()
        return response


def load_all_wav_files(directory: str = None, target_sr: int = 48000):
    # 지정 경로의 모든 WAV 음원 파일을 메모리에 일괄 로드
    import glob
    import soundfile as sf
    if directory is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sound_dir = os.path.join(base_dir, "sound")
        directory = sound_dir if os.path.exists(sound_dir) else base_dir
    wav_files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    loaded_data = []
    for filepath in wav_files:
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if data.ndim > 1:
                data = data[:, 0]
            if sr != target_sr:
                num_samples = int(len(data) * float(target_sr) / sr)
                indices = np.linspace(0, len(data) - 1, num_samples)
                data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
                sr = target_sr
            total_seconds = len(data) // sr
            if total_seconds > 0:
                loaded_data.append((os.path.basename(filepath), data, sr, total_seconds))
        except Exception as e:
            print(f"WAV 로드 오류 {filepath}: {e}", flush=True)
    return loaded_data


def ble_send(predict_result, sound_1s, sample_rate: int = 48000):
    # 20개 서브윈도우 디지털 신호 처리(DSP, Digital Signal Processing) 연산 후 164바이트 패킷을 공유 메모리에 기록
    global global_client_shm, global_seq
    try:
        if sound_1s is None or len(sound_1s) == 0:
            return
        
        if global_client_shm is None:
            try:
                global_client_shm = shared_memory.SharedMemory(name='ble_shm', create=True, size=PACKET_SIZE)
            except FileExistsError:
                global_client_shm = shared_memory.SharedMemory(name='ble_shm', create=False)
        
        status_code = 1 if (predict_result == 1 or predict_result is True) else 0
        packet = bytearray([0xAA, status_code])
        
        chunks = np.array_split(sound_1s, 20)
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            # RMS (Root Mean Square, 실효 음압 에너지) 계산
            rmse = float(np.sqrt(np.mean(chunk ** 2)))
            
            # FFT (Fast Fourier Transform, 고속 푸리에 변환) 기반 주파수 대역폭(Spectral Bandwidth) 계산
            fft_val = np.abs(np.fft.rfft(chunk))
            total_power = np.sum(fft_val) + 1e-12
            freq = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)
            center_freq = np.sum(freq * fft_val) / total_power
            bw = float(np.sqrt(np.sum(((freq - center_freq) ** 2) * fft_val) / total_power))
            
            packet.extend(struct.pack('<ff', rmse, bw))
        
        if len(packet) < 162:
            packet.extend(b'\x00' * (162 - len(packet)))
        
        global_seq = (global_seq + 1) % 65536
        packet.extend(struct.pack('<H', global_seq))
        
        global_client_shm.buf[:PACKET_SIZE] = packet[:PACKET_SIZE]
    except Exception as e:
        print(f"BLE 송신 에러: {e}", flush=True)


def start_server():
    # 블루투스 GATT 서버 및 공유 메모리 스트리밍 데몬 구동
    global global_shm

    print("블루투스 무선 서비스 초기화 및 캐시 정리 중", flush=True)
    subprocess.run("sudo rm -rf /var/lib/bluetooth/*/cache/*", shell=True, check=False)
    subprocess.run("sudo systemctl restart bluetooth", shell=True, check=False)
    time.sleep(1.0)

    subprocess.run(['sudo', 'iw', 'wlan0', 'set', 'power_save', 'off'], check=False)
    subprocess.run(['sudo', 'btmgmt', 'power', 'off'], check=False)
    subprocess.run(['sudo', 'btmgmt', 'bredr', 'off'], check=False)
    subprocess.run(['sudo', 'btmgmt', 'le', 'on'], check=False)
    subprocess.run(['sudo', 'btmgmt', 'connectable', 'on'], check=False)
    subprocess.run(['sudo', 'btmgmt', 'power', 'on'], check=False)
    subprocess.run(['sudo', 'btmgmt', 'advertising', 'on'], check=False)
    subprocess.run(['bluetoothctl', 'discoverable', 'on'], check=False)

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    srv = Service(bus, 0, SERVICE_UUID, True)
    char = Characteristic(bus, 0, CHAR_UUID, ['read', 'notify'], srv)
    srv.add_characteristic(char)

    app = Application(bus)
    app.add_service(srv)

    manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/org/bluez/hci0'), GATT_MANAGER_IFACE)
    manager.RegisterApplication(app.get_path(), {}, reply_handler=lambda: print("무선 송신 앱 등록 성공", flush=True), error_handler=lambda e: print(f"무선 송신 앱 등록 실패: {e}", flush=True))

    try:
        global_shm = shared_memory.SharedMemory(name='ble_shm', create=True, size=PACKET_SIZE)
        global_shm.buf[:PACKET_SIZE] = bytearray(PACKET_SIZE)
    except FileExistsError:
        global_shm = shared_memory.SharedMemory(name='ble_shm', create=False)

    print(f"공유 메모리({PACKET_SIZE}B) 연결 완료. 실시간 모니터링 시작", flush=True)

    last_sent_data = bytearray(PACKET_SIZE)

    def sync_memory():
        nonlocal last_sent_data
        try:
            if global_shm is not None:
                current_data = bytearray(global_shm.buf[:PACKET_SIZE])
                if len(current_data) == PACKET_SIZE and current_data[0] == 0xAA:
                    if current_data != last_sent_data:
                        last_sent_data = bytearray(current_data)
                        char.value = current_data
                        if char.notifying:
                            props = dbus.Dictionary({'Value': dbus.ByteArray(current_data)}, signature='sv')
                            char.PropertiesChanged(GATT_CHRC_IFACE, props, dbus.Array([], signature='s'))
        except Exception as e:
            print(f"실시간 전송 에러: {e}", flush=True)
        return True

    GLib.timeout_add(100, sync_memory)

    try:
        GLib.MainLoop().run()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("무선 송신 서버 종료 중", flush=True)
        try:
            manager.UnregisterApplication(app.get_path())
        except:
            pass
        if global_shm is not None:
            global_shm.close()
            try:
                global_shm.unlink()
            except:
                pass


if __name__ == "__main__":
    start_server()
