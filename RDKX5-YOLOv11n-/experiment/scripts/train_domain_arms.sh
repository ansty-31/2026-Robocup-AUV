set -e
cd /home/ansty/RDKX5/auv_vision/RDKX5-YOLOv11n-
PY=/home/ansty/anaconda3/envs/yolov8/bin/python
for ARM in B C; do
  echo "==================== ARM $ARM  $(date +%H:%M:%S) ===================="
  $PY scripts/2_train/train_yolo11n.py \
     --data experiment/data/pose_${ARM}/data.yaml --task pose \
     --weights weights/yolo11n-pose.coco.pt \
     --epochs 200 --batch 4 --imgsz 640 --workers 2 \
     --device 0 --cache ram --seed 0 \
     --project experiment/runs/domain --name A_${ARM} \
     --output-pt weights/domain_${ARM}.pt --no-repatch 2>&1 | tail -45
  echo "---- ARM $ARM 完成 $(date +%H:%M:%S) ----"
done
echo "全部完成 $(date +%H:%M:%S)"
