# MicroPython untuk micro:bit (flash lewat python.microbit.org / Mu editor)
# Driver motor (mis. L298N): IN1=pin0, IN2=pin1, IN3=pin8, IN4=pin12
from microbit import *
uart.init(baudrate=115200)

def drive(a, b, c, d):
    pin0.write_digital(a); pin1.write_digital(b)
    pin8.write_digital(c); pin12.write_digital(d)

teks = {b"F": "MAJU", b"S": "STOP", b"R": "BELOK"}
while True:
    if uart.any():
        cmd = uart.read(1)
        if cmd == b"F":   drive(1, 0, 1, 0)
        elif cmd == b"R": drive(1, 0, 0, 1)
        elif cmd == b"S": drive(0, 0, 0, 0)
        if cmd in teks:
            display.scroll(teks[cmd], wait=False)   # efek marquee
    sleep(20)
