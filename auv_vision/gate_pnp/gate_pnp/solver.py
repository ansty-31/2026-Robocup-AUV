"""Stateless planar four-corner PnP; no model, ROS, camera or cached pose."""
import cv2
import numpy as np
from .models import Observation, PnPConfig, PoseCandidate, PoseResult, Status


def _quaternion(rotation):
    # Largest-component construction remains stable near 180-degree rotations.
    r=rotation
    terms=np.array([1+r[0,0]-r[1,1]-r[2,2],1-r[0,0]+r[1,1]-r[2,2],
                    1-r[0,0]-r[1,1]+r[2,2],1+np.trace(r)])
    index=int(np.argmax(terms)); q=np.zeros(4)
    q[index]=.5*np.sqrt(max(0.,terms[index])); denominator=4*q[index]
    if index==3:
        q[:3]=[r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]]
        q[:3]/=denominator
    else:
        j,k=(index+1)%3,(index+2)%3
        q[j]=(r[j,index]+r[index,j])/denominator
        q[k]=(r[k,index]+r[index,k])/denominator
        q[3]=(r[k,j]-r[j,k])/denominator
    q/=np.linalg.norm(q)
    return -q if q[3]<0 else q


def _candidate(rvec,tvec,xyz,xy,camera,refined=False):
    rvec=np.asarray(rvec,np.float64).reshape(3)
    tvec=np.asarray(tvec,np.float64).reshape(3)
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    rotation,_=cv2.Rodrigues(rvec)
    if np.any((xyz@rotation.T+tvec)[:,2] <= 1e-9):
        return None
    projected,_=cv2.projectPoints(xyz,rvec,tvec,camera.K,camera.distortion)
    errors=np.linalg.norm(projected.reshape(-1,2)-xy,axis=1)
    rms=float(np.sqrt(np.mean(errors**2)))
    if not np.isfinite(rms):
        return None
    forward=np.eye(4);forward[:3,:3]=rotation;forward[:3,3]=tvec
    inverse=np.eye(4);inverse[:3,:3]=rotation.T;inverse[:3,3]=-rotation.T@tvec
    return PoseCandidate(forward,inverse,inverse[:3,3].copy(),rotation.T.copy(),
                         _quaternion(rotation.T),rvec,tvec,rms,errors,refined)


def _same_pose(a,b,config):
    relative=a.camera_rotation_gate@b.camera_rotation_gate.T
    angle=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
    distance=float(np.linalg.norm(a.camera_position_gate-b.camera_position_gate))
    return angle <= config.same_pose_rotation_deg and distance <= config.same_pose_translation_m


def _validate_points(obs,camera,config):
    points=np.asarray(obs.points,dtype=np.float64)
    confidence=np.asarray(obs.confidence,dtype=np.float64)
    if points.shape != (4,2) or not np.isfinite(points).all():
        raise ValueError('Exactly four finite pixel pairs TL/TR/BR/BL are required')
    if confidence.shape != (4,) or not np.isfinite(confidence).all():
        raise ValueError('Four finite keypoint confidences are required')
    if np.any((confidence<config.keypoint_confidence)|(confidence>1)):
        raise ValueError('Keypoint confidence is below threshold or outside [0,1]')
    if (not np.isfinite(obs.detection_confidence)
            or not config.detection_confidence <= obs.detection_confidence <= 1):
        raise ValueError('Detection confidence is below threshold or outside [0,1]')
    if np.any(points<0) or np.any(points >= np.array(camera.image_size)):
        raise ValueError('Keypoints lie outside the 640x640 image')
    delta=points[:,None,:]-points[None,:,:]
    distances=np.linalg.norm(delta,axis=2)+np.eye(4)*1e9
    if np.min(distances) < config.min_edge_px:
        raise ValueError('Duplicate or excessively close keypoints')
    edges=np.roll(points,-1,axis=0)-points
    following=np.roll(edges,-1,axis=0)
    cross=edges[:,0]*following[:,1]-edges[:,1]*following[:,0]
    if not (np.all(cross>1e-8) or np.all(cross<-1e-8)):
        raise ValueError('Keypoints are collinear, nonconvex or self-intersecting')
    area=.5*abs(np.sum(points[:,0]*np.roll(points[:,1],-1)
                        -points[:,1]*np.roll(points[:,0],-1)))
    if area < config.min_quad_area_px2 or area/np.max(np.sum(edges**2,axis=1)) < config.min_area_edge_ratio:
        raise ValueError('Projected gate is too small or geometrically degenerate')
    return np.ascontiguousarray(points)


def estimate_gate_pose(observation, camera_geometry, config=None):
    """Return camera-in-gate pose only for an unambiguous valid current input.

    preprocessing_id is the caller's explicit coordinate-space declaration;
    pixel data alone cannot reveal incorrect preprocessing or semantic labels.
    """
    if observation is None:
        return PoseResult(Status.NO_INPUT,'No keypoint observation supplied')
    if not isinstance(observation,Observation):
        return PoseResult(Status.INVALID_KEYPOINTS,'Expected an Observation instance')
    obs=observation
    context=dict(target_index=obs.target_index,frame_id=obs.frame_id,timestamp=obs.timestamp)
    def result(status,reason,**kwargs):
        return PoseResult(status,reason,**context,**kwargs)
    config=PnPConfig() if config is None else config
    try:
        config.validate();camera_geometry.validate()
        if (tuple(obs.image_size) != camera_geometry.image_size
                or obs.preprocessing_id != camera_geometry.preprocessing_id):
            raise ValueError('Observation size/preprocessing does not match camera geometry')
    except (AttributeError,TypeError,ValueError,KeyError) as exc:
        return result(Status.CONFIG_ERROR,str(exc))
    try:
        xy=_validate_points(obs,camera_geometry,config)
    except (ValueError,TypeError,OverflowError) as exc:
        return result(Status.INVALID_KEYPOINTS,str(exc))
    w,h=config.width_m/2,config.height_m/2
    xyz=np.array([[-w,-h,0],[w,-h,0],[w,h,0],[-w,h,0]],np.float64)
    try:
        count,rvecs,tvecs,_=cv2.solvePnPGeneric(xyz,xy,camera_geometry.K,
                            camera_geometry.distortion,flags=cv2.SOLVEPNP_IPPE)
        if not count:
            return result(Status.SOLVE_FAILED,'IPPE returned no solutions')
        candidates=[]
        for rvec,tvec in zip(rvecs,tvecs):
            initial=_candidate(rvec,tvec,xyz,xy,camera_geometry)
            if initial is None:
                continue
            best=initial
            try:
                rr,tt=cv2.solvePnPRefineLM(xyz,xy,camera_geometry.K,camera_geometry.distortion,
                          initial.rvec.reshape(3,1).copy(),initial.tvec.reshape(3,1).copy(),
                          criteria=(cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS,50,1e-10))
                refined=_candidate(rr,tt,xyz,xy,camera_geometry,True)
                if refined is not None and refined.rms_error_px <= initial.rms_error_px:
                    best=refined
            except cv2.error:
                pass  # Preserve the valid IPPE estimate if refinement fails.
            candidates.append(best)
        candidates.sort(key=lambda p:p.rms_error_px)
        unique=[]
        for candidate in candidates:
            if not any(_same_pose(candidate,other,config) for other in unique):
                unique.append(candidate)
        if not unique:
            return result(Status.SOLVE_FAILED,'No finite solution with all four points in front of camera')
        best=unique[0]
        details=dict(rms_error_px=best.rms_error_px,candidates=tuple(unique))
        if best.rms_error_px > config.max_rms_error_px:
            return result(Status.HIGH_REPROJECTION_ERROR,'Reprojection RMS exceeds configured limit',**details)
        if (len(unique)>1
                and unique[1].rms_error_px-best.rms_error_px < config.ambiguity_gap_px):
            return result(Status.AMBIGUOUS,'Distinct planar poses have similar reprojection errors',**details)
        return result(Status.SUCCESS,'Valid current-frame camera pose in gate coordinates',pose=best,**details)
    except (cv2.error,ValueError,np.linalg.LinAlgError) as exc:
        return result(Status.SOLVE_FAILED,f'PnP failed: {exc}')
