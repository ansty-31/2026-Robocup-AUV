# -*- coding: utf-8 -*-
"""手动模式相机切换：UDP 指令解析 + 切换逻辑。"""
from manual.cam_switch import CamPusher, parse_camera_packet


def test_parse_camera_packet():
    assert parse_camera_packet(b"cam:front") == "front"
    assert parse_camera_packet(b"cam:f") == "front"
    assert parse_camera_packet(b"cam:down") == "down"
    assert parse_camera_packet(b"cam:d") == "down"
    assert parse_camera_packet(b"cam") == "toggle"
    assert parse_camera_packet(b"cam:toggle") == "toggle"
    assert parse_camera_packet(b"cam:t") == "toggle"
    assert parse_camera_packet(b"cam:x") == "bad"
    # 运动包不是相机指令
    assert parse_camera_packet(b"1,0,0,0") is None
    assert parse_camera_packet(b"stop") is None


def test_cam_pusher_switch_toggle():
    p = CamPusher(None)                 # 不推流 ⇒ 不开相机（纯逻辑）
    assert p.which == "front"
    assert p.switch("down") is True
    assert p.which == "down"
    assert p.switch("side") is False    # 非法名不动
    assert p.which == "down"
    assert p.toggle() == "front"
    assert p.toggle() == "down"
    p.close()


def test_cam_toggle_sends_and_debounces(monkeypatch):
    """PC 端 Tab 键 → 发 `cam` 到板端遥控桥；0.3s 防抖（连按只切一次）。"""
    import manual.stream as st
    sent = []

    class _Sock:
        def sendto(self, data, addr):
            sent.append((data, addr))

        def close(self):
            pass

    monkeypatch.setattr(st, "socket", type("_SockMod", (), {
        "socket": lambda *a, **k: _Sock(),
        "AF_INET": 2, "SOCK_DGRAM": 2}))
    t = st.CamToggle("1.2.3.4", 9000)
    t._last = 0.0
    t.send(); t.send(); t.send()          # 0.3s 内只发一次
    assert len(sent) == 1, sent
    assert sent[0] == (b"cam", ("1.2.3.4", 9000))
    t._last = -1.0                        # 越过防抖窗
    t.send()
    assert len(sent) == 2
    t.close()
