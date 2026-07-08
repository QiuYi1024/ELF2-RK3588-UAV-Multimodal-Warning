#include "VideoWidget.h"
#include "DataManager.h"

#include <QPainter>
#include <QPainterPath>
#include <QTimer>
#include <QFontMetrics>
#include <QDateTime>
#include <QLineF>
#include <QPolygonF>
#include <QMouseEvent>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;

double normalizeAngle180(double deg) {
    while (deg > 180.0) deg -= 360.0;
    while (deg < -180.0) deg += 360.0;
    return deg;
}

double normalizedUnit(double value) {
    if (!std::isfinite(value)) return 0.0;
    if (value > 1.5) value /= 100.0;
    return qBound(0.0, value, 1.0);
}

bool isFiniteAngle(double deg) {
    return std::isfinite(deg) && std::fabs(deg) < 9000.0;
}

QRect boundedLabelRect(const QRect& preferred, const QRect& bounds) {
    QRect r = preferred;
    if (r.right() > bounds.right()) r.moveRight(bounds.right());
    if (r.left() < bounds.left()) r.moveLeft(bounds.left());
    if (r.bottom() > bounds.bottom()) r.moveBottom(bounds.bottom());
    if (r.top() < bounds.top()) r.moveTop(bounds.top());
    return r;
}

void drawAudioSourceOverlay(QPainter& painter,
                            const QRect& videoRect,
                            const QRect& visualBox,
                            bool lightPaint,
                            qint64 nowMs,
                            bool hasVisualLock) {
    if (videoRect.width() <= 40 || videoRect.height() <= 40) {
        return;
    }

    const qint64 lastAudioMs = g_lastAudioRecvMs.load();
    const bool audioFresh = lastAudioMs <= 0 || (nowMs - lastAudioMs) <= 2400;
    const bool audioDetected = g_latestAudioDetected.load();
    const bool doaValid = g_latestAudioDoaValid.load() || g_hasAudioSignal.load();
    if (!audioFresh || !doaValid) {
        return;
    }

    const double rawAngle = static_cast<double>(g_latestAudioAngle.load());
    const double panErr = g_latestAudioPanErrDeg.load();
    const bool useRelativePan = isFiniteAngle(panErr);
    const double relativeDeg = normalizeAngle180(useRelativePan ? panErr : rawAngle);
    if (!isFiniteAngle(relativeDeg)) {
        return;
    }

    const double score = normalizedUnit(g_latestAudioScoreEma.load());
    const double doaConfidence = normalizedUnit(g_latestAudioDoaConfidence.load());
    const double stability = normalizedUnit(g_latestAudioDoaStability.load());
    const bool stableDoa = g_latestAudioStableDoa.load();
    const double confidence = std::max({score, doaConfidence, stability});

    const double clippedRel = qBound(-90.0, relativeDeg, 90.0);
    const double xRatio = qBound(-1.0, clippedRel / 70.0, 1.0);
    const bool outsideView = std::fabs(relativeDeg) > 58.0;
    const bool occludedCandidate = audioDetected && !hasVisualLock;

    QColor mainColor = hasVisualLock ? QColor(45, 212, 191) : QColor(251, 191, 36);
    if (!audioDetected) {
        mainColor = QColor(56, 189, 248);
    }
    QColor lineColor = mainColor;
    lineColor.setAlpha(audioDetected ? 230 : 170);

    QPointF target;
    if (visualBox.isValid() && !visualBox.isEmpty()) {
        target = visualBox.center();
    } else {
        const int sourceX = videoRect.center().x() + static_cast<int>(xRatio * videoRect.width() * 0.44);
        const int sourceY = videoRect.y() + static_cast<int>(videoRect.height() * 0.42);
        target = QPointF(sourceX, sourceY);
    }
    target.setX(qBound(static_cast<qreal>(videoRect.left() + 26), target.x(), static_cast<qreal>(videoRect.right() - 26)));
    target.setY(qBound(static_cast<qreal>(videoRect.top() + 42), target.y(), static_cast<qreal>(videoRect.bottom() - 48)));

    painter.save();
    painter.setClipRect(videoRect.adjusted(1, 1, -1, -1));
    painter.setRenderHint(QPainter::Antialiasing, !lightPaint);

    // 参考声源定位演示图，用局部热力斑表达声源方位，避免大扇形遮挡视频。
    QPointF arrowStart = target + QPointF(-videoRect.width() * 0.18, videoRect.height() * 0.22);
    if (!hasVisualLock && outsideView) {
        arrowStart = QPointF(videoRect.center().x(), videoRect.bottom() - 42);
    }
    arrowStart.setX(qBound(static_cast<qreal>(videoRect.left() + 30), arrowStart.x(), static_cast<qreal>(videoRect.right() - 30)));
    arrowStart.setY(qBound(static_cast<qreal>(videoRect.top() + 58), arrowStart.y(), static_cast<qreal>(videoRect.bottom() - 30)));

    QLineF arrowLine(arrowStart, target);
    if (arrowLine.length() > 18.0) {
        const double a = std::atan2(arrowLine.dy(), arrowLine.dx());
        const QPointF arrowEnd = target - QPointF(std::cos(a) * 13.0, std::sin(a) * 13.0);
        const QPointF wing1 = arrowEnd - QPointF(std::cos(a - 0.55) * 12.0, std::sin(a - 0.55) * 12.0);
        const QPointF wing2 = arrowEnd - QPointF(std::cos(a + 0.55) * 12.0, std::sin(a + 0.55) * 12.0);
        painter.setPen(QPen(QColor(239, 68, 68, audioDetected ? 230 : 170), audioDetected ? 2.2 : 1.5,
                            stableDoa ? Qt::SolidLine : Qt::DashLine));
        painter.drawLine(arrowStart, arrowEnd);
        painter.drawLine(arrowEnd, wing1);
        painter.drawLine(arrowEnd, wing2);
    }

    const int radius = qBound(26,
                              static_cast<int>(std::min(videoRect.width(), videoRect.height()) *
                                               (0.052 + (1.0 - confidence) * 0.025)),
                              74);
    QRectF heatRect(target.x() - radius, target.y() - radius * 0.78,
                    radius * 2.0, radius * 1.56);

    QRadialGradient heat(target, radius);
    heat.setColorAt(0.00, QColor(239, 68, 68, audioDetected ? 220 : 160));
    heat.setColorAt(0.20, QColor(249, 115, 22, audioDetected ? 210 : 150));
    heat.setColorAt(0.40, QColor(250, 204, 21, audioDetected ? 190 : 130));
    heat.setColorAt(0.62, QColor(34, 197, 94, audioDetected ? 150 : 105));
    heat.setColorAt(0.80, QColor(56, 189, 248, audioDetected ? 120 : 80));
    heat.setColorAt(1.00, QColor(56, 189, 248, 0));
    painter.setPen(Qt::NoPen);
    painter.setBrush(heat);
    painter.drawEllipse(heatRect);

    QPolygonF contour;
    const double rx = radius * 0.95;
    const double ry = radius * 0.70;
    const double contourScale[12] = {0.84, 1.03, 0.91, 1.08, 0.88, 1.00, 0.92, 1.07, 0.86, 0.98, 0.90, 1.02};
    for (int i = 0; i < 12; ++i) {
        const double t = (static_cast<double>(i) / 12.0) * 2.0 * kPi;
        contour << QPointF(target.x() + std::cos(t) * rx * contourScale[i],
                           target.y() + std::sin(t) * ry * contourScale[(i + 3) % 12]);
    }
    painter.setPen(QPen(QColor(59, 130, 246, audioDetected ? 190 : 120), 1.5));
    painter.setBrush(Qt::NoBrush);
    painter.drawPolygon(contour);

    painter.setBrush(QColor(239, 68, 68, audioDetected ? 220 : 150));
    painter.setPen(QPen(QColor(255, 247, 237, audioDetected ? 230 : 170), 1));
    painter.drawEllipse(target, std::max(4, radius / 8), std::max(3, radius / 10));

    if (outsideView) {
        const int edgeX = relativeDeg >= 0.0 ? videoRect.right() - 18 : videoRect.left() + 18;
        const int edgeY = qBound(videoRect.top() + 56,
                                 static_cast<int>(target.y() + videoRect.height() * 0.16),
                                 videoRect.bottom() - 56);
        QPolygonF arrow;
        if (relativeDeg >= 0.0) {
            arrow << QPointF(edgeX + 10, edgeY)
                  << QPointF(edgeX - 8, edgeY - 12)
                  << QPointF(edgeX - 8, edgeY + 12);
        } else {
            arrow << QPointF(edgeX - 10, edgeY)
                  << QPointF(edgeX + 8, edgeY - 12)
                  << QPointF(edgeX + 8, edgeY + 12);
        }
        painter.setPen(Qt::NoPen);
        painter.setBrush(mainColor);
        painter.drawPolygon(arrow);
    }

    const int barW = qBound(52, static_cast<int>(videoRect.width() * 0.11), 110);
    QRect confBar(static_cast<int>(target.x()) - barW / 2,
                  static_cast<int>(target.y()) + radius + 8,
                  barW,
                  6);
    confBar = boundedLabelRect(confBar, videoRect.adjusted(16, 16, -16, -16));
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(15, 23, 42, 190));
    painter.drawRoundedRect(confBar, 3, 3);
    QRect confFill = confBar.adjusted(1, 1, -1, -1);
    confFill.setWidth(std::max(2, static_cast<int>(confFill.width() * confidence)));
    painter.setBrush(mainColor);
    painter.drawRoundedRect(confFill, 2, 2);

    painter.restore();

    QString title = audioDetected ? "声源定位" : "声源候选";
    QString relation = useRelativePan
        ? QString("相对视轴 %1°").arg(relativeDeg, 0, 'f', 1)
        : QString("DOA %1°").arg(rawAngle, 0, 'f', 1);
    QString detail;
    if (occludedCandidate) {
        detail = outsideView ? "视觉未确认 / 视野外" : "视觉未确认 / 可能遮挡";
    } else if (hasVisualLock && audioDetected) {
        detail = "音视联动";
    } else {
        detail = stableDoa ? "方向稳定" : "方向待稳定";
    }
    const QString label = QString("%1  %2  %3  置信%4%")
                              .arg(title)
                              .arg(relation)
                              .arg(detail)
                              .arg(static_cast<int>(std::round(confidence * 100.0)));

    QFont labelFont("Microsoft YaHei UI", 10, QFont::Bold);
    painter.setFont(labelFont);
    QRect labelRect = QFontMetrics(labelFont).boundingRect(label).adjusted(-10, -5, 10, 5);
    labelRect.moveTopLeft(QPoint(static_cast<int>(target.x()) - labelRect.width() / 2,
                                 static_cast<int>(target.y()) - radius - labelRect.height() - 8));
    labelRect = boundedLabelRect(labelRect, videoRect.adjusted(14, 14, -14, -14));

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(3, 7, 12, 218));
    painter.drawRoundedRect(labelRect, 4, 4);
    painter.setPen(mainColor);
    painter.drawText(labelRect, Qt::AlignCenter, label);
}
}

VideoWidget::VideoWidget(QWidget* parent) : QWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    setAutoFillBackground(false);
    refreshTimer = new QTimer(this);
    refreshTimer->setTimerType(Qt::PreciseTimer);
    connect(refreshTimer, &QTimer::timeout, this, [this]() {
        const int targetInterval = 33;
        if (refreshTimer->interval() != targetInterval) {
            refreshTimer->setInterval(targetInterval);
        }
        const qint64 nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        auto timing = std::atomic_load(&g_latestVideoTiming);
        const int frameId = timing ? timing->frame_id : -1;
        const bool hasNewFrame = frameId >= 0 && frameId != lastPaintedFrameId;
        const bool overlayDue = lastOverlayUpdateMonoMs <= 0 || nowMs - lastOverlayUpdateMonoMs >= 250;
        if (hasNewFrame || overlayDue) {
            this->update();
        }
    });
    refreshTimer->start(33);
}

static QRect fitRectKeepAspect(const QSize& src, const QRect& dst) {
    if (src.width() <= 0 || src.height() <= 0) return dst;

    double sx = static_cast<double>(dst.width()) / src.width();
    double sy = static_cast<double>(dst.height()) / src.height();
    double s = std::min(sx, sy);

    int w = static_cast<int>(src.width() * s);
    int h = static_cast<int>(src.height() * s);
    int x = dst.x() + (dst.width() - w) / 2;
    int y = dst.y() + (dst.height() - h) / 2;

    return QRect(x, y, w, h);
}

static QRectF coverSourceRectKeepAspect(const QSize& src, const QRect& dst) {
    if (src.width() <= 0 || src.height() <= 0 || dst.width() <= 0 || dst.height() <= 0) {
        return QRectF(0, 0, src.width(), src.height());
    }

    const double srcAspect = static_cast<double>(src.width()) / src.height();
    const double dstAspect = static_cast<double>(dst.width()) / dst.height();

    if (srcAspect > dstAspect) {
        const double cropW = src.height() * dstAspect;
        const double x = (src.width() - cropW) / 2.0;
        return QRectF(x, 0, cropW, src.height());
    }

    const double cropH = src.width() / dstAspect;
    const double y = (src.height() - cropH) / 2.0;
    return QRectF(0, y, src.width(), cropH);
}

void VideoWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    const qint64 paintStartMonoMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auto timing = std::atomic_load(&g_latestVideoTiming);

    QPainter painter(this);
    const bool lightPaint = g_displayEcoMode.load();
    painter.setRenderHint(QPainter::Antialiasing, !lightPaint);
    const QRect outer = this->rect();
    painter.fillRect(outer, QColor(7, 10, 14));

    QRect contentRect = outer.adjusted(1, 1, -1, -1);
    painter.setPen(QPen(QColor(65, 77, 91), 1));
    painter.setBrush(QColor(9, 13, 18));
    painter.drawRoundedRect(contentRect, 8, 8);
    contentRect = contentRect.adjusted(8, 8, -8, -8);

    // 1. 绘制海康 RTSP 视频流
    auto framePtr = std::atomic_load(&g_latestVideoFrame);

    QRect videoRect = contentRect;
    QSize srcSize(0, 0);

    if (framePtr && !framePtr->empty()) {
        srcSize = QSize(framePtr->cols, framePtr->rows);
        videoRect = fitRectKeepAspect(srcSize, contentRect);
        lastVideoRect = videoRect;

        QImage img(framePtr->data,
                   framePtr->cols,
                   framePtr->rows,
                   static_cast<int>(framePtr->step),
                   QImage::Format_RGB888);

        const bool needsFillBackground =
            videoRect.width() + 24 < contentRect.width() ||
            videoRect.height() + 24 < contentRect.height();
        if (!lightPaint && needsFillBackground) {
            // 只在留边明显时绘制弱化填充背景，减少每帧重复缩放造成的闪烁和 CPU 压力。
            painter.save();
            painter.setClipRect(contentRect);
            painter.drawImage(contentRect, img, coverSourceRectKeepAspect(srcSize, contentRect));
            painter.fillRect(contentRect, QColor(3, 7, 12, 206));
            painter.restore();
        }

        if (!lightPaint) {
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(0, 0, 0, 130));
            painter.drawRoundedRect(videoRect.adjusted(-5, -5, 5, 6), 6, 6);
        }

        painter.drawImage(videoRect, img);

        if (!lightPaint) {
            painter.setPen(QPen(QColor(103, 232, 249, 190), 1));
            painter.setBrush(Qt::NoBrush);
            painter.drawRoundedRect(videoRect.adjusted(-1, -1, 1, 1), 4, 4);

            const int cx = videoRect.center().x();
            const int cy = videoRect.center().y();
            painter.setPen(QPen(QColor(203, 213, 225, 120), 1));
            painter.drawLine(cx - 28, cy, cx - 8, cy);
            painter.drawLine(cx + 8, cy, cx + 28, cy);
            painter.drawLine(cx, cy - 28, cx, cy - 8);
            painter.drawLine(cx, cy + 8, cx, cy + 28);
            painter.drawEllipse(QPoint(cx, cy), 4, 4);

            const int overlayW = std::max(180, std::min(videoRect.width() - 28, 360));
            QRect titleRect(videoRect.x() + 14, videoRect.y() + 12, overlayW, 34);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(3, 7, 12, 190));
            painter.drawRoundedRect(titleRect, 4, 4);

            painter.setPen(QColor(103, 232, 249));
            painter.setFont(QFont("Microsoft YaHei UI", 11, QFont::Bold));
            const QString sourceLabel = g_videoUsingRkStream.load() ? "RK同帧视频" : "海康RTSP";
            painter.drawText(titleRect.adjusted(12, 0, -8, 0),
                             Qt::AlignVCenter | Qt::AlignLeft,
                             QString("%1  %2 x %3")
                                 .arg(sourceLabel)
                                 .arg(framePtr->cols)
                                 .arg(framePtr->rows));
        }
    } else {
        lastVideoRect = QRect();
        painter.setPen(QPen(QColor(51, 65, 85), 1));
        painter.setBrush(QColor(10, 15, 22));
        painter.drawRect(videoRect);

        painter.setPen(QColor(148, 163, 184));
        painter.setFont(QFont("Microsoft YaHei UI", 20, QFont::Bold));
        painter.drawText(videoRect.adjusted(0, -18, 0, 0),
                         Qt::AlignCenter,
                         g_expectRkFrameStream.load() ? "等待 RK 同帧视频流" : "等待海康威视 RTSP 视频流");

        painter.setPen(QColor(100, 116, 139));
        painter.setFont(QFont("Microsoft YaHei UI", 13));
        painter.drawText(videoRect.adjusted(0, 34, 0, 0),
                         Qt::AlignCenter,
                         g_expectRkFrameStream.load() ? "RK3588 YOLO frame stream 5010" : "CAMERA_IP  /  Streaming Channels 102");
    }

    // 2. 绘制 YOLO 视觉框
    auto metaPtr = std::atomic_load(&g_latestSensorMeta);
    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    const int maxBoxAgeMs = std::max(80, g_visionOverlayMaxAgeMs.load());
    const bool metaFresh = metaPtr && (metaPtr->recv_ms <= 0 || (nowMs - metaPtr->recv_ms) <= maxBoxAgeMs);
    QRect visualAudioBox;
    if (framePtr && metaFresh && metaPtr && metaPtr->has_detection && metaPtr->bbox_w > 0 && metaPtr->bbox_h > 0) {
        int coordW = metaPtr->coord_w > 0 ? metaPtr->coord_w : framePtr->cols;
        int coordH = metaPtr->coord_h > 0 ? metaPtr->coord_h : framePtr->rows;

        if (coordW <= 0) coordW = 640;
        if (coordH <= 0) coordH = 640;

        float scaleX = static_cast<float>(videoRect.width()) / static_cast<float>(coordW);
        float scaleY = static_cast<float>(videoRect.height()) / static_cast<float>(coordH);

        QRectF bboxF(videoRect.x() + metaPtr->bbox_x * scaleX,
                     videoRect.y() + metaPtr->bbox_y * scaleY,
                     metaPtr->bbox_w * scaleX,
                     metaPtr->bbox_h * scaleY);

        QRect bbox = bboxF.toRect();
        visualAudioBox = bbox;

        QColor boxColor = metaPtr->tracking ? QColor(110, 231, 183) : QColor(251, 191, 36);

        painter.setPen(QPen(boxColor, 3));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(bbox);

        QString label;
        if (metaPtr->conf > 0.001f) {
            label = QString("%1  置信:%2")
                        .arg(metaPtr->tracking ? "视觉跟踪" : "视觉发现")
                        .arg(metaPtr->conf, 0, 'f', 2);
        } else {
            label = metaPtr->tracking ? "视觉跟踪" : "视觉发现";
        }

        QFont labelFont("Microsoft YaHei UI", 10, QFont::Bold);
        painter.setFont(labelFont);
        QRect labelRect = QFontMetrics(labelFont).boundingRect(label).adjusted(-8, -4, 8, 4);
        labelRect.moveTopLeft(QPoint(bbox.x(), std::max(videoRect.y(), bbox.y() - labelRect.height() - 2)));

        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(4, 9, 14, 220));
        painter.drawRoundedRect(labelRect, 4, 4);

        painter.setPen(boxColor);
        painter.drawText(labelRect, Qt::AlignCenter, label);

        // 画目标中心
        QPoint center = bbox.center();
        painter.setPen(QPen(boxColor, 2));
        painter.drawLine(center.x() - 10, center.y(), center.x() + 10, center.y());
        painter.drawLine(center.x(), center.y() - 10, center.x(), center.y() + 10);
    }

    // 3. 声源定位叠加层：遮挡或视觉未确认时仍能给出声学方向。
    const bool visualOverlayLocked = metaFresh && metaPtr && metaPtr->tracking;
    drawAudioSourceOverlay(painter, videoRect, visualAudioBox, lightPaint, nowMs, visualOverlayLocked);

    // 4. 录像状态提示
    if (g_recordingActive.load()) {
        QRect recRect(videoRect.x() + 14, videoRect.y() + videoRect.height() - 44, 128, 30);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(185, 28, 28, 230));
        painter.drawRoundedRect(recRect, 4, 4);

        painter.setPen(Qt::white);
        painter.setFont(QFont("Microsoft YaHei UI", 11, QFont::Bold));
        painter.drawText(recRect, Qt::AlignCenter, "录像中  X停止");
    } else if (g_recordingRequested.load()) {
        QRect recRect(videoRect.x() + 14, videoRect.y() + videoRect.height() - 44, 150, 30);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(146, 64, 14, 230));
        painter.drawRoundedRect(recRect, 4, 4);

        painter.setPen(Qt::white);
        painter.setFont(QFont("Microsoft YaHei UI", 10, QFont::Bold));
        painter.drawText(recRect, Qt::AlignCenter, "录像准备中");
    }

    // 5. 右下角状态提示
    painter.setPen(QColor(203, 213, 225));
    painter.setFont(QFont("Microsoft YaHei UI", 9, QFont::Bold));
    QString status = QString("视觉:%1  音频:%2  原始:%3/%4  RK同步:%5")
                         .arg(metaFresh && metaPtr && metaPtr->has_detection ? "锁定" : "等待")
                         .arg(g_hasAudioSignal.load() ? "活跃" : "监听")
                         .arg(g_rawRecordingActive.load() ? "视频" : "--")
                         .arg(g_audioRawRecordingActive.load() ? "音频" : "--")
                         .arg(g_fusedRecordingActive.load() ? "开启" : "关闭");
    painter.drawText(contentRect.adjusted(0, 0, -12, -10),
                     Qt::AlignRight | Qt::AlignBottom,
                     status);

    const QByteArray latencyOverlayEnv = qgetenv("ANTI_UAV_QT_LATENCY_OVERLAY").trimmed();
    const bool showLatencyOverlay = latencyOverlayEnv.isEmpty() || latencyOverlayEnv != "0";
    if (showLatencyOverlay && timing && timing->frame_id >= 0 && videoRect.isValid()) {
        const qint64 ageMs = timing->camera_estimated_mono_ms >= 0
            ? paintStartMonoMs - timing->camera_estimated_mono_ms
            : -1;
        const QString source = g_videoUsingRkStream.load() ? "5010" : "rtsp";
        const QString line = QString("fid %1  age %2ms  drop %3  rq %4/%5  src %6  rec %7")
                                 .arg(timing->frame_id)
                                 .arg(ageMs)
                                 .arg(timing->display_dropped_old_frames)
                                 .arg(timing->recording_queue_size)
                                 .arg(timing->recording_dropped_frames)
                                 .arg(source)
                                 .arg(g_rawRecordingActive.load() ? "on" : "off");
        QFont overlayFont("Microsoft YaHei UI", 9, QFont::Bold);
        painter.setFont(overlayFont);
        QRect overlayRect = QFontMetrics(overlayFont).boundingRect(line).adjusted(-8, -4, 8, 4);
        overlayRect.moveTopRight(videoRect.topRight() + QPoint(-14, 12));
        overlayRect = boundedLabelRect(overlayRect, videoRect.adjusted(12, 12, -12, -12));
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(3, 7, 12, 205));
        painter.drawRoundedRect(overlayRect, 4, 4);
        painter.setPen(QColor(125, 211, 252));
        painter.drawText(overlayRect, Qt::AlignCenter, line);
    }

    const qint64 paintEndMonoMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    lastOverlayUpdateMonoMs = paintEndMonoMs;
    if (timing && timing->frame_id >= 0) {
        lastPaintedFrameId = timing->frame_id;
    }
    if (timing && timing->frame_id >= 0 &&
        timing->qt_publish_mono_ms <= paintStartMonoMs) {
        logVideoDisplayTiming(*timing, paintStartMonoMs, paintEndMonoMs);
    }
}

void VideoWidget::mousePressEvent(QMouseEvent* event) {
    const QPointF point =
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        event->position();
#else
        event->localPos();
#endif
    if (event->button() == Qt::LeftButton && lastVideoRect.isValid() &&
        lastVideoRect.contains(point.toPoint())) {
        const double nx = qBound(0.0,
                                 (point.x() - lastVideoRect.left()) /
                                     static_cast<double>(lastVideoRect.width()),
                                 1.0);
        const double ny = qBound(0.0,
                                 (point.y() - lastVideoRect.top()) /
                                     static_cast<double>(lastVideoRect.height()),
                                 1.0);
        emit videoPointClicked(nx, ny);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}
