// 차선 인식 & 차선 이탈 경고 (Hough Transform, 단일 파일)
//
// 구성: 1.설정  2.영상 입출력  3.ROI  4.Canny 엣지  5.전처리 파이프라인
//       6.허프 변환 / 대표 차선 / 안정화 / 이탈 경고 / 시각화  7.main
//
// 처리 순서: 전처리(roiEdges) -> detectLines -> estimateLaneLines -> LaneTracker
//            -> computeLanePosition -> DepartureWarning -> drawLaneOverlay / drawHud
#include <algorithm>
#include <cstdio>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

namespace lane {

// 1. 설정: 모든 좌표는 프레임 크기에 대한 비율(0~1)
struct BlurParams {
    int kernel = 5;      // 홀수 (짝수면 +1 보정)
    double sigma = 0.0;  // 0이면 kernel로부터 자동 계산
};

struct CannyParams {
    // 5개 클립 비교 결과 30/90이 가장 균형이 좋았다 (high:low = 3:1).
    double low = 30.0;
    double high = 90.0;
    bool adaptive = false;  // true면 프레임 밝기 중앙값 기반 자동 임계값
    double sigma = 0.33;
    int aperture = 3;
    bool l2gradient = false;
};

// 사다리꼴 ROI. y는 위에서부터, x는 왼쪽에서부터의 비율.
struct RoiParams {
    double topY = 0.60;
    double bottomY = 0.90;
    double topLeftX = 0.42;
    double topRightX = 0.50;
    double bottomLeftX = 0.1;
    double bottomRightX = 0.80;
};

inline bool operator==(const RoiParams& a, const RoiParams& b) {
    return a.topY == b.topY && a.bottomY == b.bottomY && a.topLeftX == b.topLeftX &&
           a.topRightX == b.topRightX && a.bottomLeftX == b.bottomLeftX &&
           a.bottomRightX == b.bottomRightX;
}
inline bool operator!=(const RoiParams& a, const RoiParams& b) { return !(a == b); }

struct PipelineConfig {
    BlurParams blur;
    CannyParams canny;
    RoiParams roi;
};

// 카메라 장착 위치에 맞춘 ROI 프리셋.
//   highway : clip1~3 (보닛 ~90%, 소실점 y ~0.55)
//   city    : clip4, videoplayback (보닛 ~75%, 소실점 y ~0.45)
bool applyRoiPreset(const string& name, PipelineConfig& cfg) {
    if (name == "highway") {
        cfg.roi = {0.60, 0.90, 0.41, 0.50, 0.12, 0.80};
        return true;
    }
    if (name == "city") {
        cfg.roi = {0.54, 0.75, 0.31, 0.57, 0.17, 0.75};
        return true;
    }
    return false;
}

// 2. 영상 입출력
class VideoSource {
public:
    // processWidth > 0 이면 프레임을 그 가로 크기로 축소한다 (1080p는 실시간에 부담).
    bool open(const string& path, int processWidth = 960) {
        if (!cap_.open(path)) return false;

        fps_ = cap_.get(CAP_PROP_FPS);
        if (fps_ <= 0.0) fps_ = 30.0;

        int w = static_cast<int>(cap_.get(CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap_.get(CAP_PROP_FRAME_HEIGHT));
        if (processWidth > 0 && processWidth < w) {
            h = static_cast<int>(h * static_cast<double>(processWidth) / w);
            w = processWidth;
        }
        size_ = {w, h};
        return true;
    }

    bool read(Mat& frame) {
        if (!cap_.read(raw_) || raw_.empty()) return false;
        if (raw_.size() == size_) {
            frame = raw_.clone();
        } else {
            resize(raw_, frame, size_, 0, 0, INTER_AREA);
        }
        return true;
    }

    double fps() const { return fps_; }
    Size frameSize() const { return size_; }

private:
    VideoCapture cap_;
    double fps_ = 30.0;
    Size size_;
    Mat raw_;
};

class VideoSink {
public:
    bool open(const string& path, double fps, Size size) {
        return writer_.open(path, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
    }
    void write(const Mat& frame) { writer_.write(frame); }
    bool isOpened() const { return writer_.isOpened(); }

private:
    VideoWriter writer_;
};

// 3. ROI: 사다리꼴 계산과 마스킹
// 좌하, 좌상, 우상, 우하 순서
vector<Point> makeRoiPolygon(Size s, const RoiParams& r) {
    const int topY = static_cast<int>(r.topY * s.height);
    const int botY = static_cast<int>(r.bottomY * s.height);
    return {
        {static_cast<int>(r.bottomLeftX * s.width), botY},
        {static_cast<int>(r.topLeftX * s.width), topY},
        {static_cast<int>(r.topRightX * s.width), topY},
        {static_cast<int>(r.bottomRightX * s.width), botY},
    };
}

// 같은 크기/파라미터면 마스크를 재사용한다.
class RoiMasker {
public:
    Mat apply(const Mat& img, const RoiParams& roi) {
        if (mask_.empty() || size_ != img.size() || params_ != roi) {
            size_ = img.size();
            params_ = roi;
            polygon_ = makeRoiPolygon(size_, roi);
            mask_ = Mat::zeros(size_, CV_8UC1);
            fillConvexPoly(mask_, polygon_, Scalar(255));
        }
        Mat out;
        bitwise_and(img, mask_, out);
        return out;
    }
    const vector<Point>& polygon() const { return polygon_; }

private:
    Mat mask_;
    Size size_;
    RoiParams params_;
    vector<Point> polygon_;
};

// 4. Canny 엣지 검출
void computeAdaptiveThresholds(const Mat& gray, double sigma, double& low, double& high) {
    // 256-bin 히스토그램에서 중앙값을 구한다 (전체 정렬보다 빠름).
    int hist[256] = {0};
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* p = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) ++hist[p[x]];
    }
    const int half = gray.rows * gray.cols / 2;
    int acc = 0, median = 0;
    for (; median < 256; ++median) {
        acc += hist[median];
        if (acc >= half) break;
    }
    low = max(0.0, (1.0 - sigma) * median);
    high = min(255.0, (1.0 + sigma) * median);
}

// 입력: 블러된 8비트 단일 채널. 출력: 0/255 엣지 맵. usedLow/usedHigh에 실제 임계값 반환.
Mat detectEdges(const Mat& blurredGray, const CannyParams& p, double* usedLow = nullptr,
                double* usedHigh = nullptr) {
    double low = p.low, high = p.high;
    if (p.adaptive) computeAdaptiveThresholds(blurredGray, p.sigma, low, high);
    if (low > high) swap(low, high);
    if (usedLow) *usedLow = low;
    if (usedHigh) *usedHigh = high;

    Mat edges;
    Canny(blurredGray, edges, low, high, p.aperture, p.l2gradient);
    return edges;
}

// 5. 전처리 파이프라인: BGR -> Gray -> Blur -> Canny -> ROI
//    다음 단계(허프 변환)는 PreprocessResult::roiEdges 를 입력으로 쓴다.
struct PreprocessResult {
    Mat gray;      // CV_8UC1
    Mat blurred;   // CV_8UC1
    Mat edges;     // ROI 적용 전 전체 프레임 엣지 (디버그용)
    Mat roiEdges;  // ROI 밖은 0. HoughLinesP 입력. 좌표계는 입력 프레임과 동일
    vector<Point> roiPolygon;  // 좌하, 좌상, 우상, 우하
    double cannyLow = 0.0;
    double cannyHigh = 0.0;
};

class PreprocessPipeline {
public:
    explicit PreprocessPipeline(const PipelineConfig& cfg = {}) : cfg_(cfg) {}

    // 반환 참조는 다음 process() 호출 전까지 유효하다.
    const PreprocessResult& process(const Mat& bgrFrame) {
        cvtColor(bgrFrame, result_.gray, COLOR_BGR2GRAY);

        int k = cfg_.blur.kernel;
        if (k < 1) k = 1;
        if (k % 2 == 0) ++k;
        GaussianBlur(result_.gray, result_.blurred, Size(k, k), cfg_.blur.sigma);

        // 마스크를 블러 영상에 먼저 씌우면 ROI 경계가 강한 엣지로 잡히므로,
        // 전체 프레임에서 Canny를 한 뒤 엣지 맵에 마스크를 적용한다.
        result_.edges =
            detectEdges(result_.blurred, cfg_.canny, &result_.cannyLow, &result_.cannyHigh);
        result_.roiEdges = masker_.apply(result_.edges, cfg_.roi);
        result_.roiPolygon = masker_.polygon();
        return result_;
    }

    PipelineConfig& config() { return cfg_; }

private:
    PipelineConfig cfg_;
    RoiMasker masker_;
    PreprocessResult result_;
};
// 6. 허프 변환: ROI 엣지 영상에서 직선 선분 검출
vector<Vec4i> detectLines(const Mat& roiEdges) {
    vector<Vec4i> lines;

    HoughLinesP(
        roiEdges,
        lines,
        1,              // rho
        CV_PI / 180,    // theta: 1도
        30,             // threshold
        30,             // minLineLength
        20              // maxLineGap
    );

    return lines;
}

// 허프 변환으로 검출한 선분을 영상에 표시
void drawLines(Mat& frame, const vector<Vec4i>& lines) {
    for (const Vec4i& l : lines) {
        cv::line(
            frame,
            Point(l[0], l[1]),
            Point(l[2], l[3]),
            Scalar(0, 0, 255),  // 빨간색
            3                  // 선 두께
        );
    }
}

// 차선 중앙과 영상 중앙의 위치 정보
struct LanePosition {
    bool valid = false;
    double laneCenterX = 0.0;
    double roiCenterX = 0.0;
    double offsetPx = 0.0;
    double offsetRatio = 0.0;
};
// 차선 변경 감지 상태
struct LaneChangeDetector {
    bool armed = false;       // 차선 변경 감지 준비 상태
    int centerFrames = 0;     // 중앙에 머문 프레임 수
    int deviationFrames = 0;  // 중앙에서 벗어난 프레임 수
    int alertFrames = 0;      // 알림을 유지할 프레임 수
};

// 좌우 대표 차선 (ROI 상단/하단 y에서의 x 좌표로 표현)
struct LaneLines {
    bool leftFound = false, rightFound = false;
    double leftTopX = 0.0, leftBottomX = 0.0;
    double rightTopX = 0.0, rightBottomX = 0.0;
    int topY = 0, bottomY = 0;

    bool valid() const { return leftFound && rightFound; }
};

// 허프 선분을 좌우로 분류하고 길이 가중 평균으로 대표 차선을 계산 (그리기는 하지 않음)
LaneLines estimateLaneLines(
    const vector<Vec4i>& lines,
    const vector<Point>& roiPolygon,
    int frameWidth
) {
    LaneLines lanes;

    // ROI 좌표 확인
    if (roiPolygon.size() < 2) {
        return lanes;
    }

    lanes.topY = roiPolygon[1].y;
    lanes.bottomY = roiPolygon[0].y;
    const double centerX = frameWidth / 2.0;

    double leftSlopeSum = 0.0, leftInterceptSum = 0.0;
    double rightSlopeSum = 0.0, rightInterceptSum = 0.0;

    double leftWeight = 0.0, rightWeight = 0.0;

    // 1. 허프 변환 결과를 좌우 차선으로 분류
    for (const Vec4i& l : lines) {
        double x1 = l[0], y1 = l[1];
        double x2 = l[2], y2 = l[3];

        if (abs(x2 - x1) < 1) continue;

        double slope = (y2 - y1) / (x2 - x1);

        if (abs(slope) < 0.3 || abs(slope) > 3.0) continue;

        double midpointX = (x1 + x2) / 2.0;
        double length = norm(Point2d(x2 - x1, y2 - y1));
        double intercept = y1 - slope * x1;

        if (slope < 0 && midpointX < centerX) {
            leftSlopeSum += slope * length;
            leftInterceptSum += intercept * length;
            leftWeight += length;
        }
        else if (slope > 0 && midpointX > centerX) {
            rightSlopeSum += slope * length;
            rightInterceptSum += intercept * length;
            rightWeight += length;
        }
    }

    // 특정 y 좌표에서 차선의 x 좌표 계산
    auto getX = [](double slope, double intercept, int y) {
        return (y - intercept) / slope;
    };

    // 2. 좌우 각각 검출된 쪽만 대표 차선 계산 (한쪽만 잡혀도 추적기가 이어 준다)
    if (leftWeight > 0) {
        double s = leftSlopeSum / leftWeight;
        double b = leftInterceptSum / leftWeight;
        lanes.leftTopX = getX(s, b, lanes.topY);
        lanes.leftBottomX = getX(s, b, lanes.bottomY);
        lanes.leftFound = true;
    }
    if (rightWeight > 0) {
        double s = rightSlopeSum / rightWeight;
        double b = rightInterceptSum / rightWeight;
        lanes.rightTopX = getX(s, b, lanes.topY);
        lanes.rightBottomX = getX(s, b, lanes.bottomY);
        lanes.rightFound = true;
    }
    return lanes;
}

// 프레임 간 차선 안정화: 지수이동평균(EMA) + 짧은 미검출 구간은 직전 값 유지
class LaneTracker {
public:
    double alpha = 0.25;  // 새 측정값 반영 비율 (작을수록 안정, 클수록 빠른 반응)
    int maxMiss = 10;     // 이 프레임 수까지는 못 찾아도 직전 차선 유지

    const LaneLines& update(const LaneLines& m) {
        state_.topY = m.topY;
        state_.bottomY = m.bottomY;
        updateSide(m.leftFound, m.leftTopX, m.leftBottomX,
                   state_.leftFound, state_.leftTopX, state_.leftBottomX, leftMiss_);
        updateSide(m.rightFound, m.rightTopX, m.rightBottomX,
                   state_.rightFound, state_.rightTopX, state_.rightBottomX, rightMiss_);
        return state_;
    }
    const LaneLines& state() const { return state_; }

private:
    void updateSide(bool found, double top, double bottom,
                    bool& valid, double& sTop, double& sBottom, int& miss) {
        if (found) {
            if (valid) {
                sTop = alpha * top + (1.0 - alpha) * sTop;
                sBottom = alpha * bottom + (1.0 - alpha) * sBottom;
            } else {  // 처음 잡혔거나 오래 놓쳤다가 다시 잡힘 -> 바로 대입
                sTop = top;
                sBottom = bottom;
            }
            valid = true;
            miss = 0;
        } else if (valid && ++miss > maxMiss) {
            valid = false;
        }
    }

    LaneLines state_;
    int leftMiss_ = 0, rightMiss_ = 0;
};

// 차선 중앙과 ROI 중앙(차량 위치)의 오프셋 계산
LanePosition computeLanePosition(const LaneLines& lanes, const vector<Point>& roiPolygon) {
    LanePosition position;
    if (!lanes.valid() || roiPolygon.size() < 4) {
        return position;
    }

    // ROI 하단 기준으로 차선 중앙 계산
    position.laneCenterX = (lanes.leftBottomX + lanes.rightBottomX) / 2.0;

    // ROI 하단의 좌우 경계 사이 중앙 계산 (카메라 = 차량 중앙 가정)
    position.roiCenterX = (roiPolygon[0].x + roiPolygon[3].x) / 2.0;

    // ROI 너비 계산
    double roiWidth = abs(roiPolygon[3].x - roiPolygon[0].x);
    if (roiWidth < 1) {
        return {};
    }

    // 두 중심 사이의 거리와 ROI 너비 대비 비율
    position.offsetPx = position.laneCenterX - position.roiCenterX;
    position.offsetRatio = position.offsetPx / roiWidth;
    position.valid = true;
    return position;
}

// 차선 이탈 경고
//   차선 중앙이 오른쪽(+)에 있으면 차량은 왼쪽으로 치우친 것 -> 왼쪽 이탈
//   깜빡임 방지: 기준을 넘은 상태가 onFrames 동안 이어져야 켜지고,
//               offRatio 아래로 돌아와야 꺼진다 (히스테리시스)
enum class Departure { None, Left, Right };

struct DepartureWarning {
    double onRatio = 0.10;   // |오프셋 비율| 이 이 값을 넘으면 이탈 후보
    double offRatio = 0.07;  // 이 값 아래로 돌아오면 경고 해제
    int onFrames = 3;        // 연속으로 넘어야 경고를 켜는 프레임 수

    Departure state = Departure::None;
    int count = 0;

    Departure update(const LanePosition& position) {
        // 한쪽 차선이라도 없으면 오경보를 막기 위해 판단 보류 (현재 상태 유지 안 함)
        if (!position.valid) {
            count = 0;
            state = Departure::None;
            return state;
        }

        const double r = position.offsetRatio;
        const Departure dir = r > 0 ? Departure::Left : Departure::Right;

        if (state == Departure::None) {
            count = (abs(r) > onRatio) ? count + 1 : 0;
            if (count >= onFrames) state = dir;
        } else if (abs(r) < offRatio) {
            state = Departure::None;
            count = 0;
        } else {
            state = dir;
        }
        return state;
    }
};

// 결과 화면 그리기: 주행 영역, 좌우 대표 차선, 차선 중앙/차량 위치 표시
void drawLaneOverlay(
    Mat& frame,
    const LaneLines& lanes,
    const LanePosition& position,
    Departure departure
) {
    const int topY = lanes.topY, bottomY = lanes.bottomY;
    const Point lt(cvRound(lanes.leftTopX), topY), lb(cvRound(lanes.leftBottomX), bottomY);
    const Point rt(cvRound(lanes.rightTopX), topY), rb(cvRound(lanes.rightBottomX), bottomY);

    // 1. 좌우 차선 사이 주행 영역 (평소 초록, 이탈 시 빨강) 반투명 채우기
    if (lanes.valid()) {
        Mat overlay = frame.clone();
        const vector<Point> laneArea = {lt, rt, rb, lb};
        const Scalar color = departure == Departure::None ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
        fillConvexPoly(overlay, laneArea, color, LINE_AA);
        addWeighted(overlay, 0.25, frame, 0.75, 0, frame);
    }

    // 2. 좌우 대표 차선 (좌: 빨강, 우: 파랑). 영역 위에 그려 선명하게 유지
    if (lanes.leftFound) cv::line(frame, lt, lb, Scalar(0, 0, 255), 5, LINE_AA);
    if (lanes.rightFound) cv::line(frame, rt, rb, Scalar(255, 0, 0), 5, LINE_AA);

    // 3. 흰 점: 차선 중앙 / 노란 점: 차량 위치(ROI 중앙)
    if (position.valid) {
        circle(frame, Point(cvRound(position.laneCenterX), bottomY), 7,
               Scalar(255, 255, 255), FILLED, LINE_AA);
        circle(frame, Point(cvRound(position.roiCenterX), bottomY), 7,
               Scalar(0, 255, 255), FILLED, LINE_AA);
    }
}

// 화면 상단 정보: FPS, 오프셋, 이탈 경고 배너
void drawHud(Mat& frame, double fps, const LanePosition& position, Departure departure) {
    auto text = [&](const string& s, Point org, double scale, Scalar color, int thick) {
        putText(frame, s, org, FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 0, 0), thick + 3, LINE_AA);
        putText(frame, s, org, FONT_HERSHEY_SIMPLEX, scale, color, thick, LINE_AA);
    };

    text(format("FPS: %.1f", fps), Point(20, 35), 0.8, Scalar(0, 255, 0), 2);
    if (position.valid) {
        text(format("Offset: %+.1f px (%+.1f%%)", position.offsetPx, position.offsetRatio * 100.0),
             Point(20, 70), 0.7, Scalar(0, 255, 255), 2);
    } else {
        text("Lane: searching...", Point(20, 70), 0.7, Scalar(200, 200, 200), 2);
    }

    if (departure != Departure::None) {
        const string msg = departure == Departure::Left ? "LANE DEPARTURE! <<< LEFT"
                                                        : "LANE DEPARTURE! RIGHT >>>";
        int base = 0;
        const Size ts = getTextSize(msg, FONT_HERSHEY_SIMPLEX, 1.1, 3, &base);
        const Point org((frame.cols - ts.width) / 2, 130);  // 좌상단 FPS/오프셋 글자와 겹치지 않게
        rectangle(frame, org + Point(-12, base + 10), org + Point(ts.width + 12, -ts.height - 12),
                  Scalar(0, 0, 200), FILLED);
        putText(frame, msg, org, FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
    }
}

// 차선 중심의 이동을 이용한 차선 변경 감지
bool updateLaneChangeDetector(
    const LanePosition& position,
    LaneChangeDetector& detector
) {
    // 알림 유지 시간 감소
    if (detector.alertFrames > 0) {
        detector.alertFrames--;
    }

    // 좌우 차선을 검출하지 못한 경우 감지하지 않음
    if (!position.valid) {
        detector.deviationFrames = 0;
        return false;
    }

    // 감지 기준값
    const double CENTER_THRESHOLD = 0.03;  // 중앙으로 판단할 범위
    const double CHANGE_THRESHOLD = 0.08;  // 중앙에서 벗어난 기준
    const int CENTER_REQUIRED = 10;        // 중앙에 머물러야 하는 프레임 수
    const int CHANGE_REQUIRED = 8;         // 이탈 상태가 지속되어야 하는 프레임 수

    double offset = std::abs(position.offsetRatio);

    // 1. 차선 중심이 기준 중앙에 위치한 경우
    if (offset <= CENTER_THRESHOLD) {
        detector.centerFrames++;
        detector.deviationFrames = 0;

        // 일정 시간 중앙에 있으면 감지 준비 완료
        if (detector.centerFrames >= CENTER_REQUIRED) {
            detector.armed = true;
        }
    }

    // 2. 차선 중심이 기준에서 크게 벗어난 경우
    else if (offset >= CHANGE_THRESHOLD) {
        detector.centerFrames = 0;
        detector.deviationFrames++;
    }

    // 3. 중앙과 이탈 기준 사이의 애매한 구간
    else {
        detector.centerFrames = 0;
        detector.deviationFrames = 0;
    }

    // 4. 이탈 상태가 일정 시간 지속되면 차선 변경 감지
    if (detector.armed &&
        detector.deviationFrames >= CHANGE_REQUIRED) {

        detector.armed = false;
        detector.centerFrames = 0;
        detector.deviationFrames = 0;

        // 알림 표시 시간 설정
        detector.alertFrames = 45;

        return true;
    }

    return false;
}
}  // namespace lane

// 7. main: 프레임 루프, 처리 시간 측정, 결과/디버그 화면, 데모 영상 저장, 튜닝 UI
namespace {

const char* kKeys =
    "{help h ?  |      | 도움말}"
    "{@video    |      | 입력 영상 경로}"
    "{preset    |highway| ROI 프리셋: highway | city}"
    "{width     |960   | 처리 해상도(가로). 0이면 원본}"
    "{blur      |5     | Gaussian 커널 크기(홀수)}"
    "{low       |30    | Canny 하한 임계값}"
    "{high      |90    | Canny 상한 임계값}"
    "{auto      |      | Canny 임계값 자동(중앙값 기반)}"
    "{tune      |      | 트랙바로 파라미터 실시간 조절}"
    "{out       |      | 결과 영상 저장 경로. --out=demo.mp4 형식}"
    "{debug     |      | 디버그 화면(결과+엣지)으로 시작. --out과 같이 쓰면 디버그 화면 저장}"
    "{no-display|      | 창 없이 실행(처리 속도 측정용)}"
    "{max-frames|0     | 처리할 최대 프레임 수(0=전체)}";

// 트랙바는 int에 값을 쓰므로 설정값을 정수로 들고 있다가 매 프레임 cfg에 반영한다.
struct TuneState {
    int blur = 5;
    int cannyLow = 30, cannyHigh = 90;
    int topY = 60, bottomY = 90, topL = 42, topR = 58, botL = 3, botR = 97;  // %

    void load(const lane::PipelineConfig& c) {
        blur = c.blur.kernel;
        cannyLow = static_cast<int>(c.canny.low);
        cannyHigh = static_cast<int>(c.canny.high);
        topY = static_cast<int>(c.roi.topY * 100 + 0.5);
        bottomY = static_cast<int>(c.roi.bottomY * 100 + 0.5);
        topL = static_cast<int>(c.roi.topLeftX * 100 + 0.5);
        topR = static_cast<int>(c.roi.topRightX * 100 + 0.5);
        botL = static_cast<int>(c.roi.bottomLeftX * 100 + 0.5);
        botR = static_cast<int>(c.roi.bottomRightX * 100 + 0.5);
    }
    void store(lane::PipelineConfig& c) const {
        c.blur.kernel = blur;
        c.canny.low = cannyLow;
        c.canny.high = cannyHigh;
        c.roi = {topY / 100.0, bottomY / 100.0, topL / 100.0, topR / 100.0, botL / 100.0,
                 botR / 100.0};
    }
};

void onTrackbar(int pos, void* target) { *static_cast<int*>(target) = pos; }

void addTrackbar(const char* win, const char* label, int* value, int maxValue) {
    createTrackbar(label, win, nullptr, maxValue, onTrackbar, value);
    setTrackbarPos(label, win, *value);
}

void createTuneWindow(const char* win, TuneState& t) {
    namedWindow(win, WINDOW_NORMAL);
    resizeWindow(win, 480, 420);
    addTrackbar(win, "blur k", &t.blur, 25);
    addTrackbar(win, "canny low", &t.cannyLow, 255);
    addTrackbar(win, "canny high", &t.cannyHigh, 255);
    addTrackbar(win, "roi top y%", &t.topY, 100);
    addTrackbar(win, "roi bottom y%", &t.bottomY, 100);
    addTrackbar(win, "roi top-L x%", &t.topL, 100);
    addTrackbar(win, "roi top-R x%", &t.topR, 100);
    addTrackbar(win, "roi bot-L x%", &t.botL, 100);
    addTrackbar(win, "roi bot-R x%", &t.botR, 100);
}

// 좌: 결과 화면 + ROI 외곽선, 우: ROI 적용된 엣지(허프 입력) + 허프 선분(빨강)
Mat makeDebugView(const Mat& result, const lane::PreprocessResult& r, const vector<Vec4i>& lines) {
    Mat left = result.clone();
    polylines(left, r.roiPolygon, true, Scalar(0, 255, 255), 2, LINE_AA);

    Mat right;
    cvtColor(r.roiEdges, right, COLOR_GRAY2BGR);
    polylines(right, r.roiPolygon, true, Scalar(0, 255, 255), 1, LINE_AA);
    for (const Vec4i& l : lines) {
        cv::line(right, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }

    char buf[96];
    snprintf(buf, sizeof(buf), "canny %.0f/%.0f  hough %d", r.cannyLow, r.cannyHigh,
             static_cast<int>(lines.size()));
    putText(right, buf, {10, 28}, FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2, LINE_AA);

    Mat both;
    hconcat(left, right, both);
    return both;
}

}  // namespace

int main(int argc, char** argv) {
    CommandLineParser args(argc, argv, kKeys);
    args.about("Lane detection & lane departure warning (Hough Transform)");
    if (args.has("help") || !args.has("@video")) {
        args.printMessage();
        return args.has("help") ? 0 : 1;
    }

    lane::PipelineConfig cfg;
    const string preset = args.get<string>("preset");
    if (!lane::applyRoiPreset(preset, cfg)) {
        fprintf(stderr, "알 수 없는 프리셋: %s (highway | city)\n", preset.c_str());
        return 1;
    }
    cfg.blur.kernel = args.get<int>("blur");
    cfg.canny.low = args.get<double>("low");
    cfg.canny.high = args.get<double>("high");
    cfg.canny.adaptive = args.has("auto");

    lane::VideoSource source;
    if (!source.open(args.get<string>("@video"), args.get<int>("width"))) {
        fprintf(stderr, "영상을 열 수 없습니다: %s\n", args.get<string>("@video").c_str());
        return 1;
    }

    const bool display = !args.has("no-display");
    const bool tune = display && args.has("tune");
    const int maxFrames = args.get<int>("max-frames");
    bool debugView = args.has("debug");      // d 키로 화면 전환 (저장 영상은 시작 시 모드로 고정)
    const bool saveDebug = debugView;

    // 결과 화면은 프레임 크기, 디버그 화면은 (결과 + 엣지) 가로 2배
    const Size frameSize = source.frameSize();
    const Size debugSize(frameSize.width * 2, frameSize.height);

    lane::VideoSink sink;
    if (args.has("out") &&
        !sink.open(args.get<string>("out"), source.fps(), saveDebug ? debugSize : frameSize)) {
        fprintf(stderr, "출력 파일을 만들 수 없습니다. (--out=파일.mp4 형식)\n");
        return 1;
    }

    lane::PreprocessPipeline pipeline(cfg);

    const char* kMain = "Lane detection";
    const char* kTune = "Tuning";
    TuneState tuneState;
    auto fitWindow = [&]() {
        const Size s = debugView ? debugSize : frameSize;
        const int winW = debugView ? 1400 : min(1280, s.width);
        resizeWindow(kMain, winW, winW * s.height / s.width);
    };
    if (display) {
        namedWindow(kMain, WINDOW_NORMAL);
        fitWindow();
    }
    if (tune) {
        tuneState.load(cfg);
        createTuneWindow(kTune, tuneState);
    }

    Mat frame, view;
    TickMeter procTimer, preTimer;
    double procSec = 0.0, preSec = 0.0, fps = 0.0;
    int n = 0;
    bool paused = false;

    lane::LaneTracker tracker;                     // 프레임 간 차선 안정화
    lane::DepartureWarning departureWarning;       // 차선 이탈 경고
    lane::LaneChangeDetector laneChangeDetector;   // 차선 변경 감지 (실험)
    lane::LaneLines lanes;
    lane::LanePosition position;
    lane::Departure departure = lane::Departure::None;

    while (true) {
        const bool newFrame = !paused;
        if (newFrame) {
            if (maxFrames > 0 && n >= maxFrames) break;
            if (!source.read(frame)) break;
        }

        if (tune) tuneState.store(pipeline.config());

        // ---- 처리 시간 측정 구간: 전처리 ~ 이탈 판정 (화면 표시/저장 제외) ----------
        procTimer.reset();
        procTimer.start();

        // 1) 전처리 (A)
        preTimer.reset();
        preTimer.start();
        const lane::PreprocessResult& pre = pipeline.process(frame);
        preTimer.stop();

        // 2) 허프 변환 + 좌우 대표 차선 (B)
        vector<Vec4i> lines = lane::detectLines(pre.roiEdges);
        const lane::LaneLines measured =
            lane::estimateLaneLines(lines, pre.roiPolygon, frame.cols);

        // 3) 안정화 + 오프셋 + 이탈 경고 + 차선 변경 감지
        //    일시정지 중에는 상태를 갱신하지 않는다 (같은 프레임이 반복되므로)
        if (newFrame) {
            lanes = tracker.update(measured);
            position = lane::computeLanePosition(lanes, pre.roiPolygon);
            departure = departureWarning.update(position);
            if (lane::updateLaneChangeDetector(position, laneChangeDetector)) {
                cout << "차선 변경 감지! (frame " << n << ")" << endl;
            }
        }

        procTimer.stop();
        if (newFrame) {
            procSec += procTimer.getTimeSec();
            preSec += preTimer.getTimeSec();
            const double t = procTimer.getTimeSec();
            if (t > 0) fps = (n == 0) ? 1.0 / t : 0.9 * fps + 0.1 / t;  // 지수이동평균
        }
        // ----------------------------------------------------------------------------

        // 4) 시각화 (C): 원본 frame은 그대로 두고 복사본에 그린다
        //    (일시정지 중 같은 프레임에 덧그려지거나, 그린 선이 다시 Canny에 잡히는 문제 방지)
        if (display || sink.isOpened()) {
            view = frame.clone();
            lane::drawLaneOverlay(view, lanes, position, departure);
            lane::drawHud(view, fps, position, departure);
            if (laneChangeDetector.alertFrames > 0) {
                putText(view, "LANE CHANGE DETECTED", Point(20, 105), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 0, 255), 2, LINE_AA);
            }

            if (sink.isOpened() && newFrame) {
                sink.write(saveDebug ? makeDebugView(view, pre, lines) : view);
            }
            if (display) {
                imshow(kMain, debugView ? makeDebugView(view, pre, lines) : view);
            }
        }

        if (newFrame) ++n;

        if (display) {
            const int key = waitKey(paused ? 30 : 1) & 0xFF;
            if (key == 'q' || key == 27) break;
            if (key == ' ') paused = !paused;
            if (key == 'd') {
                debugView = !debugView;
                fitWindow();
            }
        }
    }

    if (n > 0) {
        const double procMs = procSec * 1000.0 / n;
        printf("frames=%d  처리 해상도 %dx%d\n", n, frameSize.width, frameSize.height);
        printf("  전처리         %.2f ms/frame\n", preSec * 1000.0 / n);
        printf("  전체(~이탈판정) %.2f ms/frame  -> %.1f FPS\n", procMs, 1000.0 / procMs);
    }
    return 0;
}
