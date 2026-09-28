# MicroPython untuk DFRobot Maqueen (micro:bit)
from microbit import *

uart.init(baudrate=115200)
i2c.init()
ADDR = 0x10  # alamat driver motor Maqueen

def kirim(reg, v):
    arah = 0 if v >= 0 else 1      # 0 = maju, 1 = mundur
    i2c.write(ADDR, bytes([reg, arah, min(abs(v), 255)]))

def motor(kiri, kanan):
    kirim(0x00, kiri)              # motor kiri
    kirim(0x02, kanan)             # motor kanan

teks = {b"F": "MAJU", b"S": "STOP", b"R": "BELOK"}

while True:
    if uart.any():
        cmd = uart.read(1)
        if cmd == b"F":
            motor(150, 150)
        elif cmd == b"R":
            motor(150, 0)
        elif cmd == b"S":
            motor(0, 0)
        if cmd in teks:
            display.scroll(teks[cmd], wait=False)
    sleep(20)
