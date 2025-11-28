import time

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from facenet_pytorch import MTCNN

from gpio_control import setup, set_buzzer, set_lamp, cleanup

WARNING_DISTANCE_M = 6.0  # 6m 이내 접근 시 위험
CONF_THRES = 0.60         # YOLOv5 person confidence threshold


def main():
    # ---------------------------------------------------------
    # 1. 카메라 모듈 설정 및 파이프라인 설정
    # ---------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # RealSense 컬러 + 깊이 스트림 설정
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # ---------------------------------------------------------
    # 2. YOLOv5 모델 로드
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # YOLOv5s 모델 로드
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.to(device)
    model.conf = CONF_THRES  # confidence threshold

    # MTCNN 얼굴 검출기
    face_detector = MTCNN(keep_all=True, device=device)

    # GPIO 초기화
    setup()
    lamp_on = False          # 램프 현재 상태
    new_person_detected = False

    try:
        while True:
            # 프레임 수신 및 정렬
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # -------------------------------------------------
            # 3. 각 모델을 통해 객체 인식 및 얼굴 감지
            # -------------------------------------------------

            # (1) YOLOv5로 person 감지
            results = model(color_image)

            person_detected = False
            distance = -1.0

            # YOLOv5: "person" 탐지 및 거리 계산
            for *box, conf, cls in results.xyxy[0].tolist():
                if int(cls) == 0 and conf >= CONF_THRES:  # "person" 클래스는 0번 인덱스
                    x1, y1, x2, y2 = map(int, box[:4])

                    person_detected = True

                    # 중심 좌표 계산
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # 탐지된 객체의 거리 계산 (m 단위)
                    distance = depth_frame.get_distance(center_x, center_y)

                    # 경계 상자와 라벨 추가 (person)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person [conf:{conf:.2f}], [distance:{distance:.2f}m]"
                    cv2.putText(
                        color_image,
                        label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # 한 프레임에서 가장 가까운 한 명만 쓰고 싶으면 break
                    break

            # (2) MTCNN으로 얼굴 감지
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            boxes, probs = face_detector.detect(rgb_image)
            faces = boxes if boxes is not None else []
            face_detected = len(faces) > 0  # 얼굴이 감지되었는지 확인

            # 얼굴 박스 시각화 (선택)
            for box in faces:
                fx1, fy1, fx2, fy2 = map(int, box)
                cv2.rectangle(color_image, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

            # -------------------------------------------------
            # 4. 주어진 조건 만족 시 부저 및 램프 작동
            #    (스샷에 있던 로직 그대로 반영)
            # -------------------------------------------------

            # 조건 1: 0m < 거리 <= 6m 이고, 얼굴이 보이지 않을 때 → 새로운 위험 상황
            if distance > 0 and distance <= WARNING_DISTANCE_M and not face_detected:
                new_person_detected = True
            else:
                new_person_detected = False

            # 조건 2: 거리 조건 안에서 얼굴 여부에 따라 부저 제어
            if distance > 0 and distance <= WARNING_DISTANCE_M:
                if not face_detected:
                    # activate_buzzer()
                    set_buzzer(True)
                else:
                    # deactivate_buzzer()
                    set_buzzer(False)
            else:
                # 범위 밖이면 항상 부저 OFF
                set_buzzer(False)

            # 램프 상태 관리 (스샷 흐름 맞춰 정리)
            if new_person_detected and not lamp_on:
                lamp_on = True
                # control_lamp(True)
                set_lamp(True)
            elif not new_person_detected and lamp_on:
                lamp_on = False
                # control_lamp(False)
                set_lamp(False)

            # 화면 출력
            cv2.imshow("Bicycle-Pedestrian Auto Warning System", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # 안전 종료
        set_buzzer(False)
        set_lamp(False)
        cleanup()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()