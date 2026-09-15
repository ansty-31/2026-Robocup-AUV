"""Geometry for the supplied alpha=0 undistort + anisotropic resize pipeline."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from .models import PIPELINE_ID, _plain


@dataclass(frozen=True)
class CameraGeometry:
    K: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int,int]
    preprocessing_id: str
    metadata: dict

    def validate(self):
        if self.preprocessing_id != PIPELINE_ID or tuple(self.image_size) != (640,640):
            raise ValueError('Expected the approved undistorted 640x640 resize pipeline')
        k = np.asarray(self.K,dtype=float)
        d = np.asarray(self.distortion,dtype=float)
        if (k.shape != (3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1]) <= 0
                or not np.allclose(k[2],[0,0,1]) or abs(k[0,1])+abs(k[1,0]) > 1e-10):
            raise ValueError('Invalid final camera matrix')
        if d.size != 5 or not np.isfinite(d).all() or np.any(d != 0):
            raise ValueError('Already-undistorted pixels require five zero distortion coefficients')
        m = self.metadata
        if (m.get('original_size_wh') != [1280,720] or m.get('output_size_wh') != [640,640]
                or m.get('undistort_alpha') != 0 or m.get('center_principal_point') is not False
                or m.get('crop_applied') is not False or m.get('resize_mode') != 'stretch'
                or m.get('opencv_version') != cv2.__version__):
            raise ValueError('Preprocessing metadata/version does not match this runtime')
        new = np.asarray(m.get('new_camera_matrix'),dtype=float)
        expected = resize_pixel_transform() @ new if new.shape == (3,3) else None
        if expected is None or not np.allclose(k,expected,rtol=1e-10,atol=1e-10):
            raise ValueError('Final intrinsics do not match undistortion and resize metadata')

    def to_dict(self):
        return _plain({'K':self.K, 'distortion':self.distortion,'image_size':self.image_size,
                       'preprocessing_id':self.preprocessing_id,'metadata':self.metadata})

    def save(self,path):
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2,
                                        allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def from_file(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        geometry=cls(np.asarray(data['K'],float),np.asarray(data['distortion'],float),
                     tuple(data['image_size']),data['preprocessing_id'],data['metadata'])
        geometry.validate()
        return geometry


def resize_pixel_transform():
    # cv2.resize uses src=(dst+0.5)/scale-0.5 for its pixel-center sampling.
    sx,sy=640/1280,640/720
    return np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]],np.float64)


def geometry_from_files(calibration_path, vision_path):
    """Compute geometry only. Does not read images or run the attachment script.

    calibration_path is explicit; the old board model/path/mock fields are ignored.
    Use the SAME OpenCV version as image preprocessing, or pass its exported geometry.
    """
    calibration_path,vision_path=Path(calibration_path),Path(vision_path)
    with vision_path.open(encoding='utf-8') as file:
        config=yaml.safe_load(file)
    if not isinstance(config,dict):
        raise ValueError('vision.yaml must be a mapping')
    try:
        size=(config['camera']['front']['width'],config['camera']['front']['height'])
        output=config['model']['input_size']
        enabled=config['image']['undistort']
    except (KeyError,TypeError) as exc:
        raise ValueError('Missing camera dimensions, undistort flag or output size') from exc
    if size != (1280,720) or output != 640 or enabled is not True:
        raise ValueError('Requires 1280x720 input, undistort=true, output size=640')
    image=config['image']
    # Reject a different pipeline rather than silently accepting an older letterbox config.
    if (image.get('undistort_alpha',0) != 0 or image.get('resize_mode','stretch') != 'stretch'
            or image.get('center_principal_point',False) is not False
            or image.get('crop',False) not in (False,None)):
        raise ValueError('vision.yaml geometry differs from supplied prepare_frames(1).py')
    if not calibration_path.is_file():
        raise ValueError(f'Calibration file missing: {calibration_path}')
    # Python handles Unicode paths on Windows; FileStorage receives YAML text.
    calibration_text=calibration_path.read_text(encoding='utf-8-sig')
    fs=cv2.FileStorage(calibration_text,cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    try:
        if not fs.isOpened():
            raise ValueError('Cannot open camera calibration')
        k=fs.getNode('camera_matrix').mat()
        d=fs.getNode('distortion_coefficients').mat()
        wh=(int(fs.getNode('image_width').real()),int(fs.getNode('image_height').real()))
        rms_node=fs.getNode('reprojection_error')
        rms=None if rms_node.empty() else float(rms_node.real())
    finally:
        fs.release()
    if (wh != size or k is None or d is None or k.shape != (3,3) or d.size != 5
            or not np.isfinite(k).all() or not np.isfinite(d).all()
            or min(k[0,0],k[1,1]) <= 0 or not np.allclose(k[2],[0,0,1])):
        raise ValueError('Invalid or mismatched calibration resolution/K/distortion')
    new,roi=cv2.getOptimalNewCameraMatrix(k,d,size,0,size,centerPrincipalPoint=False)
    metadata={
        'original_size_wh':list(size),'output_size_wh':[640,640],
        'original_camera_matrix':k.tolist(),'original_distortion':d.reshape(-1).tolist(),
        'new_camera_matrix':new.tolist(),'undistort_alpha':0,
        'center_principal_point':False,'crop_applied':False,'unused_roi_xywh':list(map(int,roi)),
        'resize_mode':'stretch','resize_pixel_convention':'opencv_half_pixel',
        'pixel_transform':resize_pixel_transform().tolist(),'opencv_version':cv2.__version__,
        'calibration_sha256':hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        'vision_sha256':hashlib.sha256(vision_path.read_bytes()).hexdigest(),
        'calibration_rms_px':rms,'chessboard_inner_corners':[11,8],
        'chessboard_square_m':.020,'calibration_medium':'underwater_with_actual_housing_user_confirmed',
    }
    geometry=CameraGeometry(resize_pixel_transform()@new,np.zeros(5),(640,640),PIPELINE_ID,metadata)
    geometry.validate()
    return geometry
