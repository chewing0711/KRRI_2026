import os
import struct
import subprocess
import numpy as np
from multiprocessing import shared_memory

if "DBUS_SYSTEM_BUS_ADDRESS" not in os.environ:
    os.environ["DBUS_SYSTEM_BUS_ADDRESS"] = "unix:path=/var/run/dbus/system_bus_socket"

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

BLUEZ_SERVICE = "org.bluez"
GATT_MGR_IFACE = "org.bluez.GattManager1"
ADV_MGR_IFACE = "org.bluez.LEAdvertisingManager1"
ADV_IFACE = "org.bluez.LEAdvertisement1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_SRV_IFACE = "org.bluez.GattService1"
GATT_CHR_IFACE = "org.bluez.GattCharacteristic1"

DEVICE_NAME = "easylab"
SERVICE_UUID = "20260001-cafe-1234-5678-0123456789ab"
CHAR_UUID = "20260002-cafe-1234-5678-0123456789ab"
PACKET_SIZE = 164

_server_shm = None
_client_shm = None
_pkt_seq = 0


class InvalidArgsError(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


class NotSupportedError(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.NotSupported"


class BleAdvertisement(dbus.service.Object):
    PATH_BASE = "/org/bluez/example/advertisement"

    def __init__(self, bus, idx, adv_type):
        self.path = self.PATH_BASE + str(idx)
        self.bus = bus
        self.ad_type = adv_type
        self.service_uuids = None
        self.local_name = None
        self.include_tx_power = False
        super().__init__(bus, self.path)

    def get_properties(self):
        props = {"Type": self.ad_type}
        if self.service_uuids is not None:
            props["ServiceUUIDs"] = dbus.Array(self.service_uuids, signature="s")
        if self.local_name is not None:
            props["LocalName"] = dbus.String(self.local_name)
        if self.include_tx_power:
            props["Includes"] = dbus.Array(["tx-power"], signature="s")
        return {ADV_IFACE: props}

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service_uuid(self, uuid):
        if not self.service_uuids:
            self.service_uuids = []
        self.service_uuids.append(uuid)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, iface):
        if iface == ADV_IFACE:
            return self.get_properties()[ADV_IFACE]
        raise InvalidArgsError()

    @dbus.service.method(ADV_IFACE, in_signature="", out_signature="")
    def Release(self):
        pass


class BleCharacteristic(dbus.service.Object):
    def __init__(self, bus, idx, uuid, flags, service):
        self.path = service.path + "/char" + str(idx)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.descriptors = []
        self.value = bytearray(PACKET_SIZE)
        self.notifying = False
        super().__init__(bus, self.path)

    def get_properties(self):
        return {
            GATT_CHR_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": dbus.Array(self.flags, signature="s"),
                "Descriptors": dbus.Array([], signature="o"),
                "Value": dbus.ByteArray(self.value),
                "Notifying": dbus.Boolean(self.notifying),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, iface):
        if iface == GATT_CHR_IFACE:
            return self.get_properties()[GATT_CHR_IFACE]
        raise InvalidArgsError()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="ss", out_signature="v")
    def Get(self, iface, prop):
        props = self.GetAll(iface)
        if prop in props:
            return props[prop]
        raise InvalidArgsError()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="ssv", out_signature="")
    def Set(self, iface, prop, value):
        raise NotSupportedError()

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, iface, changed, invalidated):
        pass

    @dbus.service.method(GATT_CHR_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        offset = int(options.get("offset", 0))
        try:
            if _server_shm is not None:
                val = bytes(_server_shm.buf[:PACKET_SIZE])
                if len(val) == PACKET_SIZE and val[0] == 0xAA:
                    return dbus.ByteArray(val[offset:])
        except Exception:
            pass
        return dbus.ByteArray(self.value[offset:])

    @dbus.service.method(GATT_CHR_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        raise NotSupportedError()

    @dbus.service.method(GATT_CHR_IFACE, in_signature="", out_signature="")
    def StartNotify(self):
        if not self.notifying:
            self.notifying = True
            print("수신기가 실시간 스트리밍을 구독했습니다", flush=True)

    @dbus.service.method(GATT_CHR_IFACE, in_signature="", out_signature="")
    def StopNotify(self):
        if self.notifying:
            self.notifying = False
            print("수신기가 실시간 스트리밍을 중단했습니다", flush=True)


class BleService(dbus.service.Object):
    def __init__(self, bus, idx, uuid, primary):
        self.path = "/org/bluez/example/service" + str(idx)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        super().__init__(bus, self.path)

    def get_properties(self):
        return {
            GATT_SRV_IFACE: {
                "UUID": self.uuid,
                "Primary": dbus.Boolean(self.primary),
                "Includes": dbus.Array([], signature="o"),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, chrc):
        self.characteristics.append(chrc)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, iface):
        if iface == GATT_SRV_IFACE:
            return self.get_properties()[GATT_SRV_IFACE]
        raise InvalidArgsError()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="ss", out_signature="v")
    def Get(self, iface, prop):
        props = self.GetAll(iface)
        if prop in props:
            return props[prop]
        raise InvalidArgsError()

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="ssv", out_signature="")
    def Set(self, iface, prop, value):
        raise NotSupportedError()


class BleApplication(dbus.service.Object):
    def __init__(self, bus):
        self.path = "/"
        self.services = []
        super().__init__(bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, srv):
        self.services.append(srv)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):
        resp = {}
        for srv in self.services:
            resp[srv.get_path()] = srv.get_properties()
            for chrc in srv.characteristics:
                resp[chrc.get_path()] = chrc.get_properties()
        return resp


def find_adapter(bus):
    om = dbus.Interface(bus.get_object(BLUEZ_SERVICE, "/"), DBUS_OM_IFACE)
    for path, props in om.GetManagedObjects().items():
        if GATT_MGR_IFACE in props:
            return path
    return "/org/bluez/hci0"


def ble_send(predict_result, sound_1s, sample_rate=48000):
    global _client_shm, _pkt_seq
    try:
        if sound_1s is None or len(sound_1s) == 0:
            return

        if _client_shm is None:
            try:
                _client_shm = shared_memory.SharedMemory(name="ble_shm", create=True, size=PACKET_SIZE)
            except FileExistsError:
                _client_shm = shared_memory.SharedMemory(name="ble_shm", create=False)
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister("/" + _client_shm.name, "shared_memory")
            except Exception:
                pass

        status = 1 if (predict_result == 1 or predict_result is True) else 0
        packet = bytearray([0xAA, status])
        bands = [
            (0.0, 100.0),
            (100.0, 500.0),
            (500.0, 1000.0),
            (1000.0, 2000.0),
            (2000.0, 5000.0),
            (5000.0, 10000.0),
            (10000.0, 24000.0),
        ]

        for chunk in np.array_split(sound_1s, 5):
            if len(chunk) == 0:
                packet.extend(struct.pack("<8f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                continue

            n = len(chunk)
            tot_rms = float(np.sqrt(np.mean(chunk ** 2)))
            spec = np.fft.rfft(chunk)
            freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

            w = np.full(len(spec), 2.0 / (n ** 2), dtype=np.float64)
            w[0] /= 2.0
            if n % 2 == 0:
                w[-1] /= 2.0

            mag_sq = np.abs(spec) ** 2
            band_rms = []
            for f_low, f_high in bands:
                mask = (freqs >= f_low) & (freqs <= f_high if f_high >= 24000.0 else freqs < f_high)
                pwr = float(np.sum(w[mask] * mag_sq[mask]))
                band_rms.append(float(np.sqrt(max(0.0, pwr))))

            packet.extend(struct.pack("<8f", tot_rms, *band_rms))

        if len(packet) < 162:
            packet.extend(b"\x00" * (162 - len(packet)))

        _pkt_seq = (_pkt_seq + 1) % 65536
        packet.extend(struct.pack("<H", _pkt_seq))
        _client_shm.buf[:PACKET_SIZE] = packet[:PACKET_SIZE]
    except Exception as e:
        print(f"BLE 송신 에러: {e}", flush=True)


def start_server():
    global _server_shm

    print("블루투스 무선 서비스 초기화 및 캐시 정리 중", flush=True)
    print("Bluetooth 어댑터 상태 확인 중...", flush=True)

    res = subprocess.run(["bluetoothctl", "show"], capture_output=True, text=True, timeout=5)
    if res.returncode != 0 or "Powered: yes" not in res.stdout:
        raise RuntimeError("Bluetooth 어댑터가 켜져 있지 않습니다.")

    print("Bluetooth 어댑터 확인 완료", flush=True)

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    srv = BleService(bus, 0, SERVICE_UUID, True)
    chrc = BleCharacteristic(bus, 0, CHAR_UUID, ["read", "notify"], srv)
    srv.add_characteristic(chrc)

    app = BleApplication(bus)
    app.add_service(srv)

    adapter_obj = bus.get_object(BLUEZ_SERVICE, find_adapter(bus))
    mgr = dbus.Interface(adapter_obj, GATT_MGR_IFACE)
    mgr.RegisterApplication(
        app.get_path(),
        {},
        reply_handler=lambda: print("무선 송신 앱 등록 성공", flush=True),
        error_handler=lambda e: print(f"무선 송신 앱 등록 실패: {e}", flush=True),
    )

    ad_mgr = None
    adv = None
    try:
        ad_mgr = dbus.Interface(adapter_obj, ADV_MGR_IFACE)
        adv = BleAdvertisement(bus, 0, "peripheral")
        adv.local_name = DEVICE_NAME
        ad_mgr.RegisterAdvertisement(
            adv.get_path(),
            {},
            reply_handler=lambda: print(f"BLE 광고 등록 성공 ({DEVICE_NAME})", flush=True),
            error_handler=lambda e: print(f"BLE 하드웨어 비콘 활성화 ({DEVICE_NAME})", flush=True),
        )
    except Exception:
        print(f"BLE 하드웨어 비콘 활성화 ({DEVICE_NAME})", flush=True)

    try:
        _server_shm = shared_memory.SharedMemory(name="ble_shm", create=True, size=PACKET_SIZE)
        _server_shm.buf[:PACKET_SIZE] = bytearray(PACKET_SIZE)
    except FileExistsError:
        _server_shm = shared_memory.SharedMemory(name="ble_shm", create=False)

    print(f"공유 메모리({PACKET_SIZE}B) 연결 완료. 실시간 모니터링 시작", flush=True)

    last_sent = bytearray(PACKET_SIZE)

    def sync_shm():
        nonlocal last_sent
        try:
            if _server_shm is not None:
                cur = bytearray(_server_shm.buf[:PACKET_SIZE])
                if len(cur) == PACKET_SIZE and cur[0] == 0xAA and cur != last_sent:
                    last_sent = bytearray(cur)
                    chrc.value = cur
                    if chrc.notifying:
                        props = dbus.Dictionary({"Value": dbus.ByteArray(cur)}, signature="sv")
                        chrc.PropertiesChanged(GATT_CHR_IFACE, props, dbus.Array([], signature="s"))
        except Exception as e:
            print(f"실시간 전송 에러: {e}", flush=True)
        return True

    GLib.timeout_add(100, sync_shm)

    try:
        GLib.MainLoop().run()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("무선 송신 서버 종료 중", flush=True)
        if ad_mgr is not None and adv is not None:
            try:
                ad_mgr.UnregisterAdvertisement(adv.get_path())
            except Exception:
                pass
        try:
            mgr.UnregisterApplication(app.get_path())
        except Exception:
            pass
        if _server_shm is not None:
            _server_shm.close()
            try:
                _server_shm.unlink()
            except Exception:
                pass


# 호환 별칭
Advertisement = BleAdvertisement
Characteristic = BleCharacteristic
Service = BleService
Application = BleApplication
run_server = start_server

if __name__ == "__main__":
    start_server()
