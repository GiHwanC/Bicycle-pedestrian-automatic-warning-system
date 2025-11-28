"""
자전거-보행자 자동 경고 시스템 최종 통합 코드 (복원본)

구성:
- RealSense D455로 컬러/깊이 영상 수집
- YOLOv5: 사람(person) 객체 인식
- MTCNN: 얼굴 인식
- 거리 계산: 사람 bbox 기준 깊이값으로 사람과 자전거(카메라) 간 거리 추정
- 경고 조건:
    사람과의 거리가 6m 이내이고, 화면에 얼굴이 보이지 않을 때
    -> 부저: 0.5초 주기로 깜빡이듯 울림
    -> 램프: 2초 동안 켜짐
"""

from __future__ import annotations
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from facenet_pytorch import MTCNN

from distance import get_distance_m
from gpio_control import cleanup, set_buzzer, set_lamp, setup
from yolo_detector import detect_person, load_yolo

WARNING_DISTANCE_M = 6.0       # 경고를 시작할 거리 임계값
LAMP_ON_DURATION = 2.0         # 램프를 켜둘 시간 (초)
BUZZER_TOGGLE_INTERVAL = 0.5   # 부저 온/오프 토글 주기 (초)

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline_profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[RealSense] Depth scale: {depth_scale}")
    return pipeline, align


def draw_info(
    frame_bgr: np.ndarray,
    person_detections,
    distances_m,
    face_boxes,
    warning_active: bool,
    min_distance: float | None,
):
    # 사람 bbox + 거리 표시
    for det, dist in zip(person_detections, distances_m):
        x1, y1, x2, y2 = det["bbox"]
        label = f"Person {det['conf']:.2f}"
        if dist is not None:
            label += f" / {dist:.2f}m"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 얼굴 bbox 표시
    if face_boxes is not None:
        for box in face_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 전체 상태 텍스트
    status_text = "SAFE"
    color = (0, 255, 0)
    if warning_active:
        status_text = "WARNING"
        color = (0, 0, 255)

    if min_distance is not None:
        status_text += f"  min_dist={min_distance:.2f}m"

    cv2.putText(
        frame_bgr,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame_bgr


def main():
    # GPIO, RealSense, YOLO, MTCNN 초기화
    setup()
    pipeline, align = init_realsense()
    yolo_model, device = load_yolo()
    face_detector = MTCNN(keep_all=True, device=device)

    lamp_on_until = 0.0
    buzzer_on = False
    last_buzzer_toggle = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # YOLO로 사람 검출
            person_detections = detect_person(yolo_model, color_image, device)

            # 각 사람에 대해 거리 계산
            distances_m = []
            min_distance = None
            for det in person_detections:
                dist = get_distance_m(depth_frame, det["bbox"])
                distances_m.append(dist)
                if dist is not None:
                    if min_distance is None or dist < min_distance:
                        min_distance = dist

            # MTCNN으로 얼굴 검출
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            face_boxes, face_probs = face_detector.detect(rgb_image)
            face_present = False
            if face_boxes is not None and face_probs is not None:
                # 0.9 이상 신뢰도의 얼굴이 하나라도 있으면 "얼굴 있음"으로 간주
                face_present = any(
                    (p is not None) and (p >= 0.9) for p in face_probs
                )

            # 경고 조건: 가까운 사람 있고, 얼굴이 화면에 없을 때
            now = time.time()
            warning_active = (
                (min_distance is not None)
                and (min_distance <= WARNING_DISTANCE_M)
                and (not face_present)
            )

            if warning_active:
                # 램프: 2초 동안 켜두기
                if now >= lamp_on_until:
                    lamp_on_until = now + LAMP_ON_DURATION
                    set_lamp(True)

                # 부저: 0.5초 간격으로 토글
                if now - last_buzzer_toggle >= BUZZER_TOGGLE_INTERVAL:
                    buzzer_on = not buzzer_on
                    set_buzzer(buzzer_on)
                    last_buzzer_toggle = now
            else:
                # 경고 조건 해제: 램프/부저 끄기
                set_lamp(False)
                set_buzzer(False)
                lamp_on_until = 0.0
                buzzer_on = False

            # 시각화
            vis_frame = draw_info(
                color_image.copy(),
                person_detections,
                distances_m,
                face_boxes,
                warning_active,
                min_distance,
            )

            cv2.imshow("RealSense - Person and Face Detection", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        print("정리 중...")
        set_buzzer(False)
        set_lamp(False)
        cleanup()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()