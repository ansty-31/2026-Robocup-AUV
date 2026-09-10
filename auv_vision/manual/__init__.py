# -*- coding: utf-8 -*-
"""manual — 手动模式三件套：录像 recorder.py / 推流 stream.py / 遥控桥 udp_server.py

统一入口是根目录的 manual.sh：
    ./manual.sh                  # 遥控桥 + 录像 + 推流
    ./manual.sh --no-record      # 只推流 + 遥控桥
也可单独运行（支持 python3 manual/xxx.py 与 python3 -m manual.xxx 两种方式）。
"""
