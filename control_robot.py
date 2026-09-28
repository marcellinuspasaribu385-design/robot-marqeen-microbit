"""Kamera -> model -> perintah serial ke micro:bit.
Pemetaan gestur (dataset rock_paper_scissors): rock=STOP, paper=MAJU, scissors=BELOK.
Jalankan: python control_robot.py --port COM3   (Linux/Mac: /dev/ttyACM0)
"""
import argparse, json, cv2, numpy as np, serial, tensorflow as tf

MAP = {"rock": b"S", "paper": b"F", "scissors": b"R"}

ap = argparse.ArgumentParser(); ap.add_argument("--port", required=True)
a = ap.parse_args()
model = tf.keras.models.load_model("marqeen_model.keras")
names = json.load(open("labels.json"))
ser = serial.Serial(a.port, 115200, timeout=0.1)
cap, last = cv2.VideoCapture(0), None
while True:
    ok, frame = cap.read()
    if not ok: break
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (160, 160))[None].astype("float32")
    p = model.predict(img, verbose=0)[0]
    label, conf = names[int(p.argmax())], float(p.max())
    if conf > 0.8 and label != last and label in MAP:
        ser.write(MAP[label]); last = label
    cv2.putText(frame, f"{label} {conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Robot Marqeen", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release(); cv2.destroyAllWindows()
