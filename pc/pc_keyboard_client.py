# -*- coding: utf-8 -*-
"""
pc_keyboard_client.py - PC-side UDP keyboard controller for AUV testing.

This script runs on the computer, not on the RDK. It sends:
    surge,sway,heave,yaw
to remote_udp_server.py running on the RDK.

Keys:
    W / S       forward / backward
    A / D       sway left / sway right
    Up / Down   heave up / heave down
    Left / Right yaw left / yaw right
    Space       stop immediately
    Esc         quit

Example:
    python pc_keyboard_client.py --host 192.168.2.2 --port 9000 --speed 0.5
"""
import argparse
import socket
import sys
import tkinter as tk


KEY_ALIASES = {
    "w": "w",
    "s": "s",
    "a": "a",
    "d": "d",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


def axis_from_keys(positive, negative, pressed, speed):
    value = 0.0
    if positive in pressed:
        value += speed
    if negative in pressed:
        value -= speed
    return max(-1.0, min(1.0, value))


class KeyboardClient(object):
    def __init__(self, host, port, speed, interval_ms):
        self.host = host
        self.port = int(port)
        self.speed = float(speed)
        self.interval_ms = int(interval_ms)
        self.addr = (self.host, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pressed = set()
        self.last_payload = None

        self.root = tk.Tk()
        self.root.title("AUV Keyboard Remote")
        self.root.geometry("520x300")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        title = tk.Label(self.root, text="AUV Keyboard Remote",
                         font=("Arial", 18, "bold"))
        title.pack(pady=(18, 8))

        self.status = tk.Label(self.root, text="", font=("Consolas", 13),
                               justify="left")
        self.status.pack(pady=8)

        help_text = (
            "W/S: forward/backward    A/D: sway left/right\n"
            "Arrow Up/Down: heave up/down\n"
            "Arrow Left/Right: yaw left/right\n"
            "Space: stop    Esc: quit\n\n"
            "Keep this window focused while driving."
        )
        tk.Label(self.root, text=help_text, font=("Arial", 11),
                 justify="left").pack(pady=8)

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.after(self.interval_ms, self.tick)

    def normalize_key(self, event):
        name = (event.keysym or "").lower()
        return KEY_ALIASES.get(name)

    def on_key_press(self, event):
        name = (event.keysym or "").lower()
        if name == "escape":
            self.close()
            return
        if name == "space":
            self.pressed.clear()
            self.send(0.0, 0.0, 0.0, 0.0, force=True)
            return
        key = self.normalize_key(event)
        if key is not None:
            self.pressed.add(key)

    def on_key_release(self, event):
        key = self.normalize_key(event)
        if key is not None:
            self.pressed.discard(key)

    def current_motion(self):
        surge = axis_from_keys("w", "s", self.pressed, self.speed)
        sway = axis_from_keys("d", "a", self.pressed, self.speed)
        heave = axis_from_keys("up", "down", self.pressed, self.speed)
        yaw = axis_from_keys("right", "left", self.pressed, self.speed)
        return surge, sway, heave, yaw

    def send(self, surge, sway, heave, yaw, force=False):
        payload = "%.3f,%.3f,%.3f,%.3f" % (surge, sway, heave, yaw)
        if force or payload != self.last_payload:
            print("[PC] -> %s:%d  %s" % (self.host, self.port, payload))
            self.last_payload = payload
        self.sock.sendto(payload.encode("ascii"), self.addr)
        self.status.config(
            text=("target: %s:%d\n"
                  "surge=% .2f  sway=% .2f  heave=% .2f  yaw=% .2f")
            % (self.host, self.port, surge, sway, heave, yaw)
        )

    def tick(self):
        self.send(*self.current_motion())
        self.root.after(self.interval_ms, self.tick)

    def close(self):
        try:
            for _ in range(5):
                self.send(0.0, 0.0, 0.0, 0.0, force=True)
        finally:
            self.sock.close()
            self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PC keyboard controller for RDK AUV UDP remote server"
    )
    parser.add_argument("--host", required=True,
                        help="RDK IP address, for example 192.168.137.10")
    parser.add_argument("--port", type=int, default=9000,
                        help="RDK UDP port, default: 9000")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="command magnitude in [-1,1], default: 1.0")
    parser.add_argument("--interval-ms", type=int, default=50,
                        help="send interval, default: 50 ms")
    args = parser.parse_args(argv)

    speed = max(0.0, min(1.0, args.speed))
    KeyboardClient(args.host, args.port, speed, args.interval_ms).run()


if __name__ == "__main__":
    main(sys.argv[1:])
