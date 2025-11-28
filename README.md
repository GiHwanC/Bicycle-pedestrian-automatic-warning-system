# Bicycle-pedestrian-automatic-warning-system

🚲 자전거-보행자 자동 경고 시스템

RealSense D455 + 인공지능 기반 보행자 위험 감지 솔루션

프로젝트 기간: 2024.03 ~ 2024.06
사용 기술: YOLOv5, MTCNN, Intel RealSense, Python, Raspberry Pi

⸻

👥 팀 구성 및 역할

이름 역할 주요 기여
김기환 (팀장) AI·임베디드 통합 개발 프로젝트 전체 구조 설계RealSense + YOLO + MTCNN 통합 모델 개발보행자 거리 측정 알고리즘 구현위험 판단 알고리즘 개발Raspberry Pi GPIO(부저·램프) 제어 펌웨어 구성테스트 시나리오 설계 및 개선 작업
팀원 데이터 수집 & 실험 보조 자전거 주행 테스트촬영 환경 세팅 및 라벨링 도움

⸻

📘 프로젝트 개요

자전거와 보행자가 동일 보행도로를 사용하는 국내 도심 환경에서는
서로의 접근을 인지하지 못해 충돌 사고가 자주 발생한다.

본 프로젝트는 Depth 카메라 + AI 인식 + 경고 장치를 결합해
보행자가 자전거의 접근을 즉시 감지하고 경고를 받을 수 있도록 설계되었다.

⸻

🎯 해결하고자 한 문제

1. 보행자의 등(Back) 방향 문제

대부분의 충돌 사고는 보행자가 뒤를 돌아보지 못해 자전거를 인지하지 못할 때 발생한다.
• 보행자 얼굴이 보이지 않음 → 자전거 접근을 모름 → 사고 확률 증가
• 뒤에서 접근하는 자전거는 보행자가 예측하기 어려움

→ 본 시스템은 MTCNN 기반 얼굴 검출로 “보행자가 뒤를 보고 있는지”를 파악하고,
등 뒤 상황이면 위험 판단을 강화한다.

⸻

2. 거리 오판에 의한 대응 지연

시각 영상만으로는 물체와의 실제 거리를 알기 어렵다.
• 기존 카메라 기반 인식은 깊이 정보 부족
• 조명/각도/거리 변화에 따라 인식 정확도 크게 변동

→ RealSense D455의 Depth 정보로 사람 중심점의 실거리(m) 를 정밀하게 계산.

⸻

3. 보행자-자전거 혼합도로에서의 인지 부족

보행자와 자전거가 공존하지만 경고 시스템이 전무한 환경이 대부분이다.
• 자전거는 빠른 속도로 진입
• 보행자는 이어폰·전화·시야 제한으로 감지 못함

→ 부저(Buzzer) + 램프(Lamp)를 결합하여
“즉시, 직관적으로 인지 가능한 경고 시스템” 제공.

⸻

🧠 핵심 아이디어

✔ 1) 사람 인식 (YOLOv5)
• 자전거 주행 중 실시간으로 person 객체만 탐지
• 다양한 조명·거리·각도에서 안정적으로 작동하도록 설정

⸻

✔ 2) 얼굴 검출 (MTCNN)
• 사람 앞모습/뒷모습 구분
• 얼굴이 안 보이면 → 사람이 자전거 방향을 보지 않는다고 판단
• 위험 로직 가중치 반영

⸻

✔ 3) 깊이 기반 거리 계산

RealSense D455 Depth Frame 활용:
• 탐지된 bounding box 중심 주변 깊이를 평균
• 정확한 거리(m) 산출
• 6 m 이내 접근 시 위험 판단

⸻

✔ 4) 위험 감지 알고리즘

다음 두 조건 모두 만족 시 경고 작동: 1. 사람까지의 실제 거리 ≤ 6m 2. 얼굴이 검출되지 않음 (뒤돌아 있음)

→ 부저 0.5초 간격 점멸,
→ 램프 2초간 점등

⸻

✔ 5) Raspberry Pi GPIO 제어
• BUZZER_PIN / LAMP_PIN 기반 경고 장치 제어
• PWM·토글 기반 반복 경고
• ctrl+c 종료 시 GPIO 자동 정리

⸻

📡 시스템 구조

[RealSense D455] → (RGB/Depth)
↓
[YOLOv5 사람 검출]
↓
[MTCNN 얼굴 검출]
↓
[거리 계산 (Depth)]
↓
[위험 판단 알고리즘]
↓
[부저/램프 경고 (Raspberry Pi GPIO)]

<img src="docs/images/system_overview.png" width="650">

⸻

🏗 디렉토리 구조 (GitHub 기준)

/AutoWarningSystem
│
├── yolo_depth.py # 메인 실행 파일
├── yolo_detector.py # YOLO 객체 인식 모듈
├── distance.py # 거리 계산 모듈
├── gpio_control.py # RPi 부저·램프 제어
├── requirements.txt # 패키지 의존성
│
├── docs/
│ ├── 보고서.pdf
│ ├── 발표자료.pptx
│ └── images/
│ ├── system_overview.png
│ ├── algorithm_flow.png
│ └── sample_detection.png

⸻

🧪 시연 화면

RealSense 컬러 + Depth + YOLO + 얼굴 검출 + 경고 UI 표시

<img src="docs/images/sample_detection.png" width="650">

⸻

🔧 주요 코드

🔹 경고 조건 로직

warning_active = (
(min_distance is not None)
and (min_distance <= WARNING_DISTANCE_M)
and (not face_present)
)

🔹 부저 토글

if now - last_buzzer_toggle >= BUZZER_TOGGLE_INTERVAL:
buzzer_on = not buzzer_on
set_buzzer(buzzer_on)
last_buzzer_toggle = now

🔹 램프 2초 점등

if now >= lamp_on_until:
lamp_on_until = now + LAMP_ON_DURATION
set_lamp(True)

⸻

🧪 테스트 환경
• Intel RealSense D455
• Raspberry Pi 4 / 5
• Python 3.10
• PyTorch 2.1
• YOLOv5s
• facenet-pytorch
• pyrealsense2

⸻

📦 설치 방법

git clone https://github.com/GiHwanC/AutoWarningSystem.git
cd AutoWarningSystem
pip install -r requirements.txt
python3 yolo_depth.py

라즈베리파이5는 PyTorch 수동 설치 필요(README 하단 참고).

⸻

🎉 프로젝트 성과
• 2024 영남대학교 AI/SW 트랙 산학 프로젝트 성과공유회 – 대상 수상
• 보행자 인식 정확도 50% → 84% 개선
• 실제 촬영·테스트 환경에서 안정적인 위험 경고 작동 검증 완료

⸻

📘 참고 자료
• 발표자료(PPT) → /docs/발표자료.pptx￼
• 최종보고서 → /docs/최종보고서.pdf￼

⸻