# Lane Detection (C++ / OpenCV)

도로 주행 영상에서 실시간으로 차선을 검출하는 과제. 이 폴더에는 **1단계: 영상 전처리 & ROI 파이프라인**이 구현되어 있다.

```
영상 → [전처리: Gray → Blur → Canny → ROI 마스크] → (허프 변환) → (차선 이탈 경고) → 표시/저장
        ^^^^^^^^^ 구현 완료 ^^^^^^^^^^^^^^^^^^^^^    ^팀원 구현 예정^
```

## 파일 구조

모든 코드는 `lane_detection.cpp` 한 파일에 있고, 번호 붙은 섹션 주석(`// 1. 설정` 등)으로 구분한다. `std`와 `cv`는 `using namespace`로 접두사를 생략했고, 알고리즘 코드는 `namespace lane` 안에 있다.

| 섹션 | 내용 |
|---|---|
| 1. 설정 | `BlurParams`, `CannyParams`, `RoiParams`, `PipelineConfig`, ROI 프리셋(`applyRoiPreset`) |
| 2. 영상 입출력 | `VideoSource`(읽기 + 처리 해상도로 축소), `VideoSink`(저장) |
| 3. ROI | `makeRoiPolygon`, `RoiMasker`(마스크 캐싱) |
| 4. Canny | `detectEdges`, `computeAdaptiveThresholds` |
| 5. 전처리 파이프라인 | `PreprocessPipeline`, `PreprocessResult` |
| 6. main | 프레임 루프, 디버그 화면, 튜닝 UI |

파일 하나를 여러 명이 고치면 git 충돌이 나기 쉬우니, 각자 자기 섹션만 수정하고 푸시 전에 `pull`로 합치는 것이 좋다.

## 다음 단계 담당자를 위한 인터페이스

`PreprocessPipeline::process(frame)`이 돌려주는 `PreprocessResult` 중 허프 변환 입력은 **`roiEdges`** 하나다.

- `roiEdges` : `CV_8UC1`, 엣지=255, ROI 밖=0. `cv::HoughLinesP`에 그대로 넣으면 된다.
- 좌표계는 `main`의 `frame`(처리 해상도로 축소된 프레임, 기본 960×540)과 동일하다. 검출한 직선은 이 프레임 위에 바로 그리면 된다.
- `roiPolygon` : ROI 꼭짓점(좌하, 좌상, 우상, 우하). 소실점/차선 폭 계산에 쓸 수 있다.
- 이탈 경고 단계는 `frame.size()`의 중앙 x를 차량 위치로 보면 된다 (카메라가 차량 중앙에 장착되었다는 가정).

연결 위치는 `main()` 안의 `다음 단계 연결 지점` 주석이다. 새 단계는 5번 섹션 뒤(`}  // namespace lane` 앞)에 `namespace lane` 안의 함수로 추가하고 `main()`에서 호출한다. 전처리 코드는 수정할 필요가 없다.

## 빌드 (Windows)

OpenCV 설치 경로의 `lib` 폴더를 `OpenCV_DIR`로 지정해야 한다 (예: `.../opencv/build/install/lib`).

```bash
cmake -S . -B build -A x64 -DOpenCV_DIR=<opencv>/build/install/lib
cmake --build build --config Release
```

Visual Studio에서는 폴더를 열면 `CMakePresets.json`(+ 개인 경로용 `CMakeUserPresets.json`)으로 자동 구성된다. 실행 시 `opencv_world*.dll`이 있는 `<opencv>/build/install/bin`이 `PATH`에 있어야 한다.

## 실행

```bash
build/Release/lane_detection.exe <영상경로> [옵션]
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--preset` | `highway` | ROI 프리셋. `highway`(clip1~3), `city`(clip4, videoplayback) |
| `--width` | `960` | 처리 해상도 가로 (0 = 원본 1920) |
| `--blur` | `5` | Gaussian 커널 크기(홀수) |
| `--low` / `--high` | `30` / `90` | Canny 임계값 |
| `--auto` | 꺼짐 | 프레임 밝기 중앙값 기반 자동 임계값 |
| `--tune` | 꺼짐 | 트랙바로 blur/Canny/ROI 실시간 조절 |
| `--out` | 없음 | 디버그 화면(원본+ROI / ROI 엣지)을 mp4로 저장 |
| `--no-display` | 꺼짐 | 창 없이 실행 (속도 측정용) |
| `--max-frames` | `0` | 앞부분만 처리 |

키: `q`/`ESC` 종료, `Space` 일시정지.

## 파라미터 선정 근거

- **처리 해상도 960×540** : 원본 1080p 대비 픽셀 4배 감소. 전처리는 프레임당 약 2 ms (clip1~4, videoplayback에서 1.7~2.0 ms) 로 실시간 여유가 크다.
- **Grayscale + Gaussian 5×5, σ 자동** : 블러 없음/3/5/7/9를 Canny 30/90으로 비교했다 (ROI 내 엣지 픽셀 수).

  | 커널 | 없음 | 3 | 5 | 7 | 9 |
  |---|---|---|---|---|---|
  | clip1 | 2768 | 2412 | 1704 | 1213 | 1142 |
  | clip3 | 1841 | 1713 | 1514 | 1461 | 1235 |
  | clip4 | 6163 | 4819 | 3709 | 3030 | 2766 |

  3→5에서 잡음성 엣지가 크게 줄고, 7 이상에서는 clip1의 먼 곳 가는 차선과 clip4의 일부 차선이 짧아지거나 사라졌다. 정답 데이터가 없어 육안 비교이며, 5를 균형점으로 선택했다.
- **Canny 30/90 (1:3)** : 5개 클립의 같은 프레임에서 50/150, 30/90, 20/60, 자동(중앙값)을 비교. 50/150은 어두운 clip3에서 차선을 놓치고, 20/60은 가드레일·차량 엣지가 과다했다.
- **ROI는 Canny 이후에 적용** : 블러 영상에 마스크를 먼저 씌우면 ROI 경계선이 강한 엣지로 잡힌다.
- **ROI 프리셋** : 카메라 장착 높이가 클립마다 달라 두 가지로 나눴다 (비율 기반이라 해상도와 무관).

  | | topY | bottomY | top x | bottom x |
  |---|---|---|---|---|
  | highway | 0.60 | 0.90 | 0.42 ~ 0.58 | 0.03 ~ 0.97 |
  | city | 0.52 | 0.75 | 0.36 ~ 0.64 | 0.00 ~ 1.00 |

  `bottomY`는 보닛/대시보드가 시작되는 높이, `topY`는 소실점 약간 아래다. 새 영상에서는 `--tune`으로 맞춘 뒤 `applyRoiPreset()`에 값을 추가하면 된다.

## 알려진 한계

- ROI는 고정 사다리꼴이다. 차선 변경 중에는 옆 차선이 ROI 밖으로 나갈 수 있다.
- city 영상은 가드레일/횡단보도/차량 엣지가 ROI 안에 남는다. 허프 단계에서 기울기 필터링이 필요하다.
