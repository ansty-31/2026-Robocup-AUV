auv_vision为rdk端(上位机)的代码
AUV为stm32(f405)代码
pc为水面PC端脚本(键盘遥控/本地录像，见 pc/README.md)

前视 USB 相机画面推流（端到端约 60~90 ms）：见 `auv_vision/doc/前视USB相机低延迟推流方案.md`
（板端 `auv_vision/manual.sh` 推流；PC 端 `pc/pc.sh` 遥控 + 录像。零转码 MJPEG 直通，
注意 UVC 相机被单进程独占，推流要与识别共用同一路采集）。