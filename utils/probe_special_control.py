"""
Probe CameraSpecialControl for temperature-related codes.

Usage (with camera connected):
    conda run -n camstim python utils/probe_special_control.py

The script opens the first available camera, then sweeps a range of
dwCtrlCode values calling CameraSpecialControl with a data buffer.
It reports any call that:
  - returns 0 (success), AND
  - leaves a value in the buffer that looks like a plausible temperature
    (interpreted as int32, uint32, float32, or float64).
"""

import sys
import ctypes
import struct
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
import mvsdk2024 as mvsdk

PLAUSIBLE_TEMP_MIN = 10.0   # °C
PLAUSIBLE_TEMP_MAX = 90.0   # °C

def looks_like_temp(buf: bytes) -> list[tuple[str, float]]:
    hits = []
    if len(buf) >= 4:
        v_i32, = struct.unpack_from('<i', buf)
        if PLAUSIBLE_TEMP_MIN <= v_i32 <= PLAUSIBLE_TEMP_MAX:
            hits.append(('int32', float(v_i32)))
        v_u32, = struct.unpack_from('<I', buf)
        if PLAUSIBLE_TEMP_MIN <= v_u32 <= PLAUSIBLE_TEMP_MAX:
            hits.append(('uint32', float(v_u32)))
        v_f32, = struct.unpack_from('<f', buf)
        if PLAUSIBLE_TEMP_MIN <= v_f32 <= PLAUSIBLE_TEMP_MAX:
            hits.append(('float32', v_f32))
    if len(buf) >= 8:
        v_f64, = struct.unpack_from('<d', buf)
        if PLAUSIBLE_TEMP_MIN <= v_f64 <= PLAUSIBLE_TEMP_MAX:
            hits.append(('float64', v_f64))
    return hits

def main():
    mvsdk.CameraSdkInit(0)
    devs = mvsdk.CameraEnumerateDevice()
    if not devs:
        print("No camera found.")
        return

    dev = devs[0]
    print(f"Opening: {dev.acFriendlyName.decode(errors='replace')}")
    hCamera = mvsdk.CameraInit(dev, -1, -1)
    mvsdk.CameraPlay(hCamera)

    buf_size = 64
    buf = (ctypes.c_uint8 * buf_size)()

    # Probe range: 0x0000–0x00FF (common vendor command space)
    # and 0x1000–0x10FF (extended range seen in some MindVision docs)
    ranges = list(range(0x0000, 0x0100)) + list(range(0x1000, 0x1100))

    print(f"Probing {len(ranges)} control codes...\n")
    hits = []

    for code in ranges:
        ctypes.memset(buf, 0, buf_size)
        ret = mvsdk._sdk.CameraSpecialControl(
            hCamera, ctypes.c_uint(code), ctypes.c_uint(0),
            ctypes.cast(buf, ctypes.c_void_p)
        )
        if ret == 0:
            raw = bytes(buf)
            temps = looks_like_temp(raw)
            hex_preview = raw[:8].hex()
            if temps:
                print(f"  [HIT] code=0x{code:04X}  ret=0  raw={hex_preview}  interp={temps}")
                hits.append((code, temps, hex_preview))
            else:
                print(f"  [OK ] code=0x{code:04X}  ret=0  raw={hex_preview} (no plausible temp)")

    mvsdk.CameraUnInit(hCamera)

    print(f"\nDone. {len(hits)} possible temperature code(s) found.")
    if hits:
        print("\nSummary:")
        for code, temps, raw in hits:
            print(f"  code=0x{code:04X}  {temps}  raw={raw}")

if __name__ == '__main__':
    main()
