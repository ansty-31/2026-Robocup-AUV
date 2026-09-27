import cv2
import numpy as np

from temporal_quad_tracker import TemporalQuadTracker, TrackerConfig


def make_frame(shift_x: int) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    corners = np.array(
        [[180 + shift_x, 120], [420 + shift_x, 120],
         [420 + shift_x, 360], [180 + shift_x, 360]],
        dtype=np.int32,
    )
    cv2.polylines(frame, [corners.reshape(-1, 1, 2)], True, (255, 255, 255), 6)
    return frame


def detect_rectangle(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    points = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
    if len(points) != 4:
        return None
    points = points.reshape(4, 2).astype(np.float32)
    top = np.argsort(points[:, 1])[:2]
    bottom = np.argsort(points[:, 1])[-2:]
    top = top[np.argsort(points[top, 0])]
    bottom = bottom[np.argsort(points[bottom, 0])[::-1]]
    return np.concatenate((points[top], points[bottom])).astype(np.float32)


def main() -> None:
    enabled = True

    def detector(image):
        return detect_rectangle(image) if enabled else None

    tracker = TemporalQuadTracker(
        detector,
        config=TrackerConfig(
            max_tracking_misses=3,
            min_area_fraction=0.001,
            max_backtrack_error_px=3.0,
        ),
    )

    assert tracker.update(make_frame(0)).valid
    assert tracker.update(make_frame(10)).valid
    enabled = False
    assert tracker.update(make_frame(20)).mode == "tracked"
    assert tracker.update(make_frame(30)).mode == "tracked"
    assert tracker.update(make_frame(40)).mode == "tracked"
    assert not tracker.update(make_frame(50)).valid
    print("smoke test passed")


if __name__ == "__main__":
    main()
