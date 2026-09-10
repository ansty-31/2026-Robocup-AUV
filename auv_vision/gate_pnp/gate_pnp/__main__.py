"""Configuration check only: never loads weights, images or sample keypoints."""
import argparse
import json
import time
from pathlib import Path
from . import (PnPConfig, PoseResult, Status, geometry_from_files, estimate_gate_pose,
               GuidanceConfig, GateGuidance)


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='Check gate PnP configuration without model or keypoints')
    parser.add_argument('--calibration',type=Path,default=root/'config/front_camera.yaml')
    parser.add_argument('--vision',type=Path,default=root/'config/vision.yaml')
    parser.add_argument('--pnp-config',type=Path,default=root/'config/pnp.json')
    parser.add_argument('--export-geometry',type=Path,help='Explicitly save derived camera geometry JSON')
    parser.add_argument('--guidance',action='store_true',help='Check decision configuration; no input returns HOLD')
    parser.add_argument('--guidance-config',type=Path,default=root/'config/guidance.json')
    args=parser.parse_args()
    exit_code=0
    try:
        config=PnPConfig.from_file(args.pnp_config)
        geometry=geometry_from_files(args.calibration,args.vision)
        if args.export_geometry:
            geometry.save(args.export_geometry)
        if args.guidance:
            guidance_config=GuidanceConfig.from_file(args.guidance_config)
            result=GateGuidance(geometry,guidance_config,config).update(None,time.monotonic())
        else:
            result=estimate_gate_pose(None,geometry,config)
    except Exception as exc:
        result=PoseResult(Status.CONFIG_ERROR,str(exc))
        exit_code=2
    print(json.dumps(result.to_dict(),ensure_ascii=True,indent=2,allow_nan=False))
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
