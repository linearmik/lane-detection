// 차선 인식 & 차선 이탈 경고 (Hough Transform)
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

const char* kCodeVersion = "r10 no-flip";

namespace lane {

struct BlurParams {
    int kernel = 5;
    double sigma = 0.0;
};

struct CannyParams {
    double low = 30.0;
    double high = 90.0;
    bool adaptive = false;
    double sigma = 0.33;
    int aperture = 3;
    bool l2gradient = false;
};

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

struct LaneFeatureParams {
    bool enabled = true;
    double ridgeOffset = 0.015;
    int whiteMin = 25;
    bool blue = true;
    int blueMin = 15;
    bool yellow = true;
    int yellowMin = 12;
    int darkTol = 8;
};

struct PipelineConfig {
    BlurParams blur;
    CannyParams canny;
    RoiParams roi;
    LaneFeatureParams feature;
};

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

class VideoSource {
public:
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

void computeAdaptiveThresholds(const Mat& gray, double sigma, double& low, double& high) {
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

struct PreprocessResult {
    Mat gray;
    Mat blurred;
    Mat edges;
    Mat feature;
    Mat roiEdges;
    vector<Point> roiPolygon;
    double cannyLow = 0.0;
    double cannyHigh = 0.0;
};

class PreprocessPipeline {
public:
    explicit PreprocessPipeline(const PipelineConfig& cfg = {}) : cfg_(cfg) {}

    const PreprocessResult& process(const Mat& bgrFrame) {
        cvtColor(bgrFrame, result_.gray, COLOR_BGR2GRAY);

        int k = cfg_.blur.kernel;
        if (k < 1) k = 1;
        if (k % 2 == 0) ++k;
        GaussianBlur(result_.gray, result_.blurred, Size(k, k), cfg_.blur.sigma);

        if (cfg_.feature.enabled) {
            makeLaneFeature(bgrFrame, k);
            result_.edges =
                detectEdges(result_.feature, cfg_.canny, &result_.cannyLow, &result_.cannyHigh);
        } else {
            result_.feature = result_.blurred;
            result_.edges =
                detectEdges(result_.blurred, cfg_.canny, &result_.cannyLow, &result_.cannyHigh);
        }
        result_.roiEdges = masker_.apply(result_.edges, cfg_.roi);
        result_.roiPolygon = masker_.polygon();
        return result_;
    }

    PipelineConfig& config() { return cfg_; }

private:
    // max(0, min(I(x) - I(x-d), I(x) - I(x+d)))
    static Mat ridge(const Mat& src, int d) {
        Mat padded;
        copyMakeBorder(src, padded, 0, 0, d, d, BORDER_REPLICATE);
        const Mat leftN = padded(Rect(0, 0, src.cols, src.rows));
        const Mat rightN = padded(Rect(2 * d, 0, src.cols, src.rows));
        Mat a, b, out;
        subtract(src, leftN, a);
        subtract(src, rightN, b);
        cv::min(a, b, out);
        return out;
    }

    void makeLaneFeature(const Mat& bgrFrame, int blurK) {
        const LaneFeatureParams& p = cfg_.feature;
        const int d = max(2, static_cast<int>(p.ridgeOffset * bgrFrame.cols));

        threshold(ridge(result_.blurred, d), result_.feature, p.whiteMin - 1, 255, THRESH_BINARY);

        if (!p.blue && !p.yellow) return;

        Mat padded, nbMin, darker, notDark;
        copyMakeBorder(result_.blurred, padded, 0, 0, d, d, BORDER_REPLICATE);
        cv::min(padded(Rect(0, 0, bgrFrame.cols, bgrFrame.rows)),
                padded(Rect(2 * d, 0, bgrFrame.cols, bgrFrame.rows)), nbMin);
        subtract(nbMin, result_.blurred, darker);
        threshold(darker, notDark, p.darkTol, 255, THRESH_BINARY_INV);
        Mat ch[3];
        split(bgrFrame, ch);

        auto addColor = [&](const Mat& colorMap, int minValue) {
            Mat blurred, bin;
            GaussianBlur(colorMap, blurred, Size(blurK, blurK), cfg_.blur.sigma);
            threshold(ridge(blurred, d), bin, minValue - 1, 255, THRESH_BINARY);
            bitwise_and(bin, notDark, bin);
            bitwise_or(result_.feature, bin, result_.feature);
        };
        if (p.blue) {
            Mat diff;
            subtract(ch[0], ch[2], diff);
            addColor(diff, p.blueMin);
        }
        if (p.yellow) {
            Mat rg, diff;
            cv::min(ch[1], ch[2], rg);
            subtract(rg, ch[0], diff);
            addColor(diff, p.yellowMin);
        }
    }

    PipelineConfig cfg_;
    RoiMasker masker_;
    PreprocessResult result_;
};
vector<Vec4i> detectLines(const Mat& roiEdges) {
    vector<Vec4i> lines;

    HoughLinesP(
        roiEdges,
        lines,
        1,
        CV_PI / 180,
        30,
        30,
        20
    );

    return lines;
}

void drawLines(Mat& frame, const vector<Vec4i>& lines) {
    for (const Vec4i& l : lines) {
        cv::line(
            frame,
            Point(l[0], l[1]),
            Point(l[2], l[3]),
            Scalar(0, 0, 255),
            3
        );
    }
}

struct LanePosition {
    bool valid = false;
    double laneCenterX = 0.0;
    double carCenterX = 0.0;
    double laneWidthPx = 0.0;
    double offsetPx = 0.0;
    double offsetRatio = 0.0;
    double laneOffsetRatio = 0.0;
};
struct LaneChangeDetector {
    bool armed = false;
    int centerFrames = 0;
    int deviationFrames = 0;
    int alertFrames = 0;
};

struct LaneLines {
    bool leftFound = false, rightFound = false;
    double leftTopX = 0.0, leftBottomX = 0.0;
    double rightTopX = 0.0, rightBottomX = 0.0;
    int topY = 0, bottomY = 0;
    bool leftInferred = false, rightInferred = false;

    bool valid() const { return leftFound && rightFound; }
};

struct LaneSegment {
    double slope, intercept, xb, length;
};

bool pickLane(vector<LaneSegment>& segs, bool isLeft, double groupGapPx, double minSupport,
              double innerLimit, const double* priorXb, double gatePx,
              double& slope, double& intercept) {
    if (segs.empty()) return false;
    sort(segs.begin(), segs.end(),
         [](const LaneSegment& a, const LaneSegment& b) { return a.xb < b.xb; });

    const double dir = isLeft ? 1.0 : -1.0;
    bool found = false;
    double bestInner = 0.0;
    size_t bestBegin = 0, bestEnd = 0;
    for (size_t i = 0; i < segs.size();) {
        size_t j = i + 1;
        while (j < segs.size() && segs[j].xb - segs[j - 1].xb <= groupGapPx) ++j;

        double len = 0.0, xbSum = 0.0;
        for (size_t t = i; t < j; ++t) {
            len += segs[t].length;
            xbSum += segs[t].xb * segs[t].length;
        }
        const double groupXb = xbSum / len;
        const double inner = dir * groupXb;
        const bool supported = len >= minSupport;
        const bool notTooInner = inner <= dir * innerLimit;
        const bool notJumpOut = !priorXb || inner >= dir * *priorXb - gatePx;
        if (supported && notTooInner && notJumpOut && (!found || inner > bestInner)) {
            found = true;
            bestInner = inner;
            bestBegin = i;
            bestEnd = j;
        }
        i = j;
    }
    if (!found) return false;

    double w = 0.0, sSum = 0.0, bSum = 0.0;
    for (size_t t = bestBegin; t < bestEnd; ++t) {
        sSum += segs[t].slope * segs[t].length;
        bSum += segs[t].intercept * segs[t].length;
        w += segs[t].length;
    }
    slope = sSum / w;
    intercept = bSum / w;
    return true;
}

LaneLines estimateLaneLines(
    const vector<Vec4i>& lines,
    const vector<Point>& roiPolygon,
    int frameWidth,
    double carX,
    const LaneLines* prior = nullptr
) {
    LaneLines lanes;

    if (roiPolygon.size() < 2) {
        return lanes;
    }

    lanes.topY = roiPolygon[1].y;
    lanes.bottomY = roiPolygon[0].y;
    const double centerX = frameWidth / 2.0;
    const double sideMargin = frameWidth * 0.08;

    vector<LaneSegment> leftSegs, rightSegs;

    for (const Vec4i& l : lines) {
        double x1 = l[0], y1 = l[1];
        double x2 = l[2], y2 = l[3];

        if (abs(x2 - x1) < 1) continue;

        double slope = (y2 - y1) / (x2 - x1);

        if (abs(slope) < 0.45 || abs(slope) > 3.0) continue;

        double midpointX = (x1 + x2) / 2.0;
        double length = norm(Point2d(x2 - x1, y2 - y1));
        double intercept = y1 - slope * x1;
        double xb = (lanes.bottomY - intercept) / slope;
        double xt = (lanes.topY - intercept) / slope;

        if (slope < 0 && midpointX < centerX && xt <= centerX + sideMargin) {
            leftSegs.push_back({slope, intercept, xb, length});
        }
        else if (slope > 0 && midpointX > centerX && xt >= centerX - sideMargin) {
            rightSegs.push_back({slope, intercept, xb, length});
        }
    }

    auto getX = [](double slope, double intercept, int y) {
        return (y - intercept) / slope;
    };

    const double groupGap = frameWidth * 0.03;
    const double minSupport = frameWidth * 0.04;
    const double innerMargin = frameWidth * 0.06;
    const double gate = frameWidth * 0.08;
    const double* priorLeft =
        (prior && prior->leftFound && !prior->leftInferred) ? &prior->leftBottomX : nullptr;
    const double* priorRight =
        (prior && prior->rightFound && !prior->rightInferred) ? &prior->rightBottomX : nullptr;
    double s = 0.0, b = 0.0;

    if (pickLane(leftSegs, true, groupGap, minSupport, carX - innerMargin, priorLeft, gate, s,
                 b)) {
        lanes.leftTopX = getX(s, b, lanes.topY);
        lanes.leftBottomX = getX(s, b, lanes.bottomY);
        lanes.leftFound = true;
    }
    if (pickLane(rightSegs, false, groupGap, minSupport, carX + innerMargin, priorRight, gate, s,
                 b)) {
        lanes.rightTopX = getX(s, b, lanes.topY);
        lanes.rightBottomX = getX(s, b, lanes.bottomY);
        lanes.rightFound = true;
    }

    if (lanes.valid() &&
        (lanes.leftTopX >= lanes.rightTopX || lanes.leftBottomX >= lanes.rightBottomX)) {
        lanes.leftFound = lanes.rightFound = false;
    }
    return lanes;
}

class LaneTracker {
public:
    double alpha = 0.25;
    int maxMiss = 15;

    const LaneLines& update(LaneLines m) {
        if (m.valid()) {
            const double wTop = m.rightTopX - m.leftTopX;
            const double wBottom = m.rightBottomX - m.leftBottomX;
            if (widthKnown_) {
                widthTop_ = 0.1 * wTop + 0.9 * widthTop_;
                widthBottom_ = 0.1 * wBottom + 0.9 * widthBottom_;
            } else {
                widthTop_ = wTop;
                widthBottom_ = wBottom;
                widthKnown_ = true;
            }
        }
        if (widthKnown_ && m.leftFound && !m.rightFound) {
            m.rightTopX = m.leftTopX + widthTop_;
            m.rightBottomX = m.leftBottomX + widthBottom_;
            m.rightFound = m.rightInferred = true;
        } else if (widthKnown_ && m.rightFound && !m.leftFound) {
            m.leftTopX = m.rightTopX - widthTop_;
            m.leftBottomX = m.rightBottomX - widthBottom_;
            m.leftFound = m.leftInferred = true;
        }

        state_.topY = m.topY;
        state_.bottomY = m.bottomY;
        state_.leftInferred = m.leftInferred;
        state_.rightInferred = m.rightInferred;
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
            } else {
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
    bool widthKnown_ = false;
    double widthTop_ = 0.0, widthBottom_ = 0.0;
};

LanePosition computeLanePosition(const LaneLines& lanes, const vector<Point>& roiPolygon,
                                 double carX) {
    LanePosition position;
    if (!lanes.valid() || roiPolygon.size() < 4) {
        return position;
    }

    position.laneCenterX = (lanes.leftBottomX + lanes.rightBottomX) / 2.0;
    position.laneWidthPx = lanes.rightBottomX - lanes.leftBottomX;
    position.carCenterX = carX;

    double roiWidth = abs(roiPolygon[3].x - roiPolygon[0].x);
    if (roiWidth < 1) {
        return {};
    }

    if (position.laneWidthPx < roiWidth * 0.3 || position.laneWidthPx > roiWidth * 1.5) {
        return {};
    }

    position.offsetPx = position.laneCenterX - position.carCenterX;
    position.offsetRatio = position.offsetPx / roiWidth;
    position.laneOffsetRatio = position.offsetPx / position.laneWidthPx;
    position.valid = true;
    return position;
}

struct CarCenterCalibrator {
    int needFrames = 60;
    double follow = 1.0 / 300;
    bool fixed = false;
    int count = 0;
    double sum = 0.0;
    double carX = -1.0;

    bool done() const { return carX >= 0.0; }

    void add(const LanePosition& position, bool warning) {
        if (!position.valid) return;
        if (!done()) {
            sum += position.laneCenterX;
            if (++count >= needFrames) carX = sum / count;
        } else if (!fixed && !warning) {
            carX += follow * (position.laneCenterX - carX);
        }
    }
};

enum class Departure { None, Left, Right };

struct DepartureWarning {
    double onRatio = 0.20;
    double offRatio = 0.14;
    int onFrames = 8;

    Departure state = Departure::None;
    int count = 0;
    bool needRecenter = false;

    bool active() const { return state != Departure::None || needRecenter; }

    Departure update(const LanePosition& position) {
        if (!position.valid) {
            if (state != Departure::None) needRecenter = true;
            count = 0;
            state = Departure::None;
            return state;
        }

        const double r = position.laneOffsetRatio;
        const Departure dir = r > 0 ? Departure::Left : Departure::Right;

        if (needRecenter) {
            if (abs(r) < offRatio) needRecenter = false;
            count = 0;
            state = Departure::None;
            return state;
        }

        if (state == Departure::None) {
            count = (abs(r) > onRatio) ? count + 1 : 0;
            if (count >= onFrames) state = dir;
        } else if (abs(r) < offRatio) {
            state = Departure::None;
            count = 0;
        } else if (dir != state) {
            state = Departure::None;
            count = 0;
            needRecenter = true;
        }
        return state;
    }
};

void drawLaneOverlay(
    Mat& frame,
    const LaneLines& lanes,
    const LanePosition& position,
    Departure departure
) {
    const int topY = lanes.topY, bottomY = lanes.bottomY;
    const Point lt(cvRound(lanes.leftTopX), topY), lb(cvRound(lanes.leftBottomX), bottomY);
    const Point rt(cvRound(lanes.rightTopX), topY), rb(cvRound(lanes.rightBottomX), bottomY);

    if (lanes.valid()) {
        Mat overlay = frame.clone();
        const vector<Point> laneArea = {lt, rt, rb, lb};
        const Scalar color = departure == Departure::None ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
        fillConvexPoly(overlay, laneArea, color, LINE_AA);
        addWeighted(overlay, 0.25, frame, 0.75, 0, frame);
    }

    if (lanes.leftFound)
        cv::line(frame, lt, lb, Scalar(0, 0, 255), lanes.leftInferred ? 2 : 5, LINE_AA);
    if (lanes.rightFound)
        cv::line(frame, rt, rb, Scalar(255, 0, 0), lanes.rightInferred ? 2 : 5, LINE_AA);

    if (position.valid) {
        circle(frame, Point(cvRound(position.laneCenterX), bottomY), 7,
               Scalar(255, 255, 255), FILLED, LINE_AA);
        circle(frame, Point(cvRound(position.carCenterX), bottomY), 7,
               Scalar(0, 255, 255), FILLED, LINE_AA);
    }
}

void drawHud(Mat& frame, double fps, const LanePosition& position, Departure departure,
             const CarCenterCalibrator& calib) {
    auto text = [&](const string& s, Point org, double scale, Scalar color, int thick) {
        putText(frame, s, org, FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 0, 0), thick + 3, LINE_AA);
        putText(frame, s, org, FONT_HERSHEY_SIMPLEX, scale, color, thick, LINE_AA);
    };

    text(format("FPS: %.1f", fps), Point(20, 35), 0.8, Scalar(0, 255, 0), 2);
    putText(frame, kCodeVersion, Point(frame.cols - 170, frame.rows - 12), FONT_HERSHEY_SIMPLEX,
            0.5, Scalar(200, 200, 200), 1, LINE_AA);
    if (!calib.done()) {
        text(format("Calibrating car center... %d/%d", calib.count, calib.needFrames),
             Point(20, 70), 0.7, Scalar(200, 200, 200), 2);
    } else if (position.valid) {
        text(format("Offset: %+.0f px (%+.0f%% of lane)", position.offsetPx,
                    position.laneOffsetRatio * 100.0),
             Point(20, 70), 0.7, Scalar(0, 255, 255), 2);
    } else {
        text("Lane: searching...", Point(20, 70), 0.7, Scalar(200, 200, 200), 2);
    }

    if (departure != Departure::None) {
        const string msg = departure == Departure::Left ? "LANE DEPARTURE! <<< LEFT"
                                                        : "LANE DEPARTURE! RIGHT >>>";
        int base = 0;
        const Size ts = getTextSize(msg, FONT_HERSHEY_SIMPLEX, 1.1, 3, &base);
        const Point org((frame.cols - ts.width) / 2, 175);
        rectangle(frame, org + Point(-12, base + 10), org + Point(ts.width + 12, -ts.height - 12),
                  Scalar(0, 0, 200), FILLED);
        putText(frame, msg, org, FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
    }
}

bool updateLaneChangeDetector(
    const LanePosition& position,
    LaneChangeDetector& detector
) {
    if (detector.alertFrames > 0) {
        detector.alertFrames--;
    }

    if (!position.valid) {
        detector.deviationFrames = 0;
        return false;
    }

    const double CENTER_THRESHOLD = 0.10;
    const double CHANGE_THRESHOLD = 0.30;
    const int CENTER_REQUIRED = 10;
    const int CHANGE_REQUIRED = 6;

    double offset = std::abs(position.laneOffsetRatio);

    if (offset <= CENTER_THRESHOLD) {
        detector.centerFrames++;
        detector.deviationFrames = 0;

        if (detector.centerFrames >= CENTER_REQUIRED) {
            detector.armed = true;
        }
    }

    else if (offset >= CHANGE_THRESHOLD) {
        detector.centerFrames = 0;
        detector.deviationFrames++;
    }

    else {
        detector.centerFrames = 0;
        detector.deviationFrames = 0;
    }

    if (detector.armed &&
        detector.deviationFrames >= CHANGE_REQUIRED) {

        detector.armed = false;
        detector.centerFrames = 0;
        detector.deviationFrames = 0;

        detector.alertFrames = 45;

        return true;
    }

    return false;
}
}

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
    "{car-x     |-1    | 차량 중심 x 비율(0~1). 생략하면 처음 2초 동안 자동 보정}"
    "{no-blue   |      | 파란 차선 채널 끄기 (비교용)}"
    "{no-yellow |      | 노란 차선 채널 끄기 (비교용)}"
    "{no-feature|      | 차선 특징 영상(ridge) 끄고 흑백 영상에 바로 Canny (비교용)}"
    "{no-display|      | 창 없이 실행(처리 속도 측정용)}"
    "{max-frames|0     | 처리할 최대 프레임 수(0=전체)}";

struct TuneState {
    int blur = 5;
    int cannyLow = 30, cannyHigh = 90;
    int topY = 60, bottomY = 90, topL = 42, topR = 58, botL = 3, botR = 97;

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

Mat makeDebugView(const Mat& result, const lane::PreprocessResult& r, const vector<Vec4i>& lines) {
    Mat left = result.clone();
    polylines(left, r.roiPolygon, true, Scalar(0, 255, 255), 2, LINE_AA);

    Mat roiMask = Mat::zeros(r.roiEdges.size(), CV_8UC1);
    fillConvexPoly(roiMask, r.roiPolygon, Scalar(255));
    Mat maskInRoi = Mat::zeros(r.roiEdges.size(), CV_8UC1);
    r.feature.copyTo(maskInRoi, roiMask);
    Mat right;
    cvtColor(maskInRoi / 2 + r.roiEdges / 2, right, COLOR_GRAY2BGR);
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

}

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
    cfg.feature.enabled = !args.has("no-feature");
    cfg.feature.blue = !args.has("no-blue");
    cfg.feature.yellow = !args.has("no-yellow");

    lane::VideoSource source;
    if (!source.open(args.get<string>("@video"), args.get<int>("width"))) {
        fprintf(stderr, "영상을 열 수 없습니다: %s\n", args.get<string>("@video").c_str());
        return 1;
    }

    const bool display = !args.has("no-display");
    const bool tune = display && args.has("tune");
    const int maxFrames = args.get<int>("max-frames");
    bool debugView = args.has("debug");
    const bool saveDebug = debugView;

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

    lane::LaneTracker tracker;
    lane::DepartureWarning departureWarning;
    lane::LaneChangeDetector laneChangeDetector;
    lane::CarCenterCalibrator calib;
    lane::LaneLines lanes;
    lane::LanePosition position;
    lane::Departure departure = lane::Departure::None;

    const double carXRatio = args.get<double>("car-x");
    if (carXRatio >= 0.0 && carXRatio <= 1.0) {
        calib.carX = carXRatio * frameSize.width;
        calib.fixed = true;
    }

    while (true) {
        const bool newFrame = !paused;
        if (newFrame) {
            if (maxFrames > 0 && n >= maxFrames) break;
            if (!source.read(frame)) break;
        }

        if (tune) tuneState.store(pipeline.config());

        procTimer.reset();
        procTimer.start();

        preTimer.reset();
        preTimer.start();
        const lane::PreprocessResult& pre = pipeline.process(frame);
        preTimer.stop();

        const double carX = calib.done()
                                ? calib.carX
                                : (pre.roiPolygon[0].x + pre.roiPolygon[3].x) / 2.0;
        vector<Vec4i> lines = lane::detectLines(pre.roiEdges);
        const lane::LaneLines measured =
            lane::estimateLaneLines(lines, pre.roiPolygon, frame.cols, carX, &tracker.state());

        if (newFrame) {
            lanes = tracker.update(measured);
            position = lane::computeLanePosition(lanes, pre.roiPolygon, carX);
            departure = calib.done() ? departureWarning.update(position) : lane::Departure::None;
            calib.add(position, departureWarning.active());
            if (lane::updateLaneChangeDetector(position, laneChangeDetector)) {
                cout << "차선 변경 감지! (frame " << n << ")" << endl;
            }
        }

        procTimer.stop();
        if (newFrame) {
            procSec += procTimer.getTimeSec();
            preSec += preTimer.getTimeSec();
            const double t = procTimer.getTimeSec();
            if (t > 0) fps = (n == 0) ? 1.0 / t : 0.9 * fps + 0.1 / t;
        }

        if (display || sink.isOpened()) {
            view = frame.clone();
            lane::drawLaneOverlay(view, lanes, position, departure);
            lane::drawHud(view, fps, position, departure, calib);
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
            if (key == 's') {
                const string base = format("snap_%05d", n);
                imwrite(base + "_raw.png", frame);
                if (!view.empty()) {
                    imwrite(base + "_result.png", view);
                    imwrite(base + "_debug.png", makeDebugView(view, pre, lines));
                }
                printf("saved %s_raw/_result/_debug.png\n", base.c_str());
            }
        }
    }

    if (n > 0) {
        const double procMs = procSec * 1000.0 / n;
        printf("frames=%d  처리 해상도 %dx%d\n", n, frameSize.width, frameSize.height);
        printf("  전처리         %.2f ms/frame\n", preSec * 1000.0 / n);
        printf("  전체(~이탈판정) %.2f ms/frame  -> %.1f FPS\n", procMs, 1000.0 / procMs);
        if (calib.done()) {
            printf("  차량 중심 x = %.0f px (화면 가로의 %.2f)  -> 다음부터 --car-x=%.2f 로 고정 가능\n",
                   calib.carX, calib.carX / frameSize.width, calib.carX / frameSize.width);
        }
    }
    return 0;
}
