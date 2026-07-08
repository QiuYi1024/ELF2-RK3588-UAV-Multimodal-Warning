#include "RadarWidget.h"
#include "DataManager.h"
#include <QPainter>
#include <QTimer>
#include <QDateTime>
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

RadarWidget::RadarWidget(QWidget* parent) : QWidget(parent) {
    QTimer* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this]() { this->update(); });
    timer->start(30); 
}

void RadarWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    int w = this->width();
    int h = this->height();
    int cx = w / 2;
    int cy = h / 2 + 12;
    int radius = std::max(40, std::min(w, h - 48) / 2 - 20);

    QRect panel = this->rect().adjusted(1, 1, -1, -1);
    painter.fillRect(this->rect(), QColor(8, 11, 15));
    painter.setPen(QPen(QColor(65, 77, 91), 1));
    painter.setBrush(QColor(17, 24, 32));
    painter.drawRoundedRect(panel, 8, 8);

    QRect titleRect = panel.adjusted(12, 10, -12, -panel.height() + 36);
    painter.setPen(QColor(226, 232, 240));
    painter.setFont(QFont("Microsoft YaHei UI", 14, QFont::Bold));
    painter.drawText(titleRect, Qt::AlignLeft | Qt::AlignVCenter, "声学雷达");

    painter.setPen(QColor(148, 163, 184));
    painter.setFont(QFont("Microsoft YaHei UI", 11));
    painter.drawText(titleRect, Qt::AlignRight | Qt::AlignVCenter, "麦克风阵列方位");

    // 画雷达底盘：弱网格 + 方位标记，方便和摄像头视角对齐。
    painter.setPen(QPen(QColor(71, 85, 105, 150), 1));
    painter.setBrush(QColor(9, 15, 20));
    painter.drawEllipse(QPoint(cx, cy), radius, radius);
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(QPoint(cx, cy), radius * 2 / 3, radius * 2 / 3);
    painter.drawEllipse(QPoint(cx, cy), radius / 3, radius / 3);

    painter.setPen(QPen(QColor(100, 116, 139, 160), 1));
    painter.drawLine(cx - radius, cy, cx + radius, cy);
    painter.drawLine(cx, cy - radius, cx, cy + radius);
    painter.drawLine(cx - radius * 7 / 10, cy - radius * 7 / 10,
                     cx + radius * 7 / 10, cy + radius * 7 / 10);
    painter.drawLine(cx - radius * 7 / 10, cy + radius * 7 / 10,
                     cx + radius * 7 / 10, cy - radius * 7 / 10);

    painter.setPen(QColor(203, 213, 225));
    painter.setFont(QFont("Microsoft YaHei UI", 9, QFont::Bold));
    painter.drawText(QRect(cx - 16, cy - radius - 20, 32, 18), Qt::AlignCenter, "0");
    painter.drawText(QRect(cx + radius + 3, cy - 9, 32, 18), Qt::AlignLeft | Qt::AlignVCenter, "90");
    painter.drawText(QRect(cx - 20, cy + radius + 2, 40, 18), Qt::AlignCenter, "180");
    painter.drawText(QRect(cx - radius - 36, cy - 9, 32, 18), Qt::AlignRight | Qt::AlignVCenter, "270");

    // 如果听到无人机声音，画出指向红线和波束
    if (g_hasAudioSignal.load()) {
        float angle = g_latestAudioAngle.load();
        
        // Qt 坐标系: 0度在3点钟，顺时针。需要将 0度映射到 12点钟 (减去90度)
        float rad = (angle - 90.0f) * kPi / 180.0f;
        int targetX = cx + static_cast<int>(radius * std::cos(rad));
        int targetY = cy + static_cast<int>(radius * std::sin(rad));

        painter.setBrush(QColor(248, 113, 113, 55));
        painter.setPen(Qt::NoPen);
        painter.drawPie(cx - radius, cy - radius, radius * 2, radius * 2,
                        static_cast<int>((-angle + 90 + 18) * 16), -36 * 16);

        // 画锁定红线和目标点。
        painter.setPen(QPen(QColor(248, 113, 113), 3));
        painter.drawLine(cx, cy, targetX, targetY);

        painter.setBrush(QColor(248, 113, 113));
        painter.setPen(QPen(QColor(254, 226, 226), 2));
        painter.drawEllipse(QPoint(targetX, targetY), 6, 6);

        painter.setPen(QColor(254, 226, 226));
        painter.setFont(QFont("Microsoft YaHei UI", 12, QFont::Bold));
        painter.drawText(panel.adjusted(14, 0, -14, -12),
                         Qt::AlignLeft | Qt::AlignBottom,
                         QString("目标方位：%1°").arg(angle, 0, 'f', 1));
    } else {
        const qint64 now = QDateTime::currentMSecsSinceEpoch();
        const float sweepAngle = std::fmod(static_cast<float>(now) / 18.0f, 360.0f);
        const float sweepRad = (sweepAngle - 90.0f) * kPi / 180.0f;
        int sweepX = cx + static_cast<int>(radius * std::cos(sweepRad));
        int sweepY = cy + static_cast<int>(radius * std::sin(sweepRad));

        painter.setPen(QPen(QColor(34, 211, 238, 160), 2));
        painter.drawLine(cx, cy, sweepX, sweepY);

        painter.setPen(QColor(148, 163, 184));
        painter.setFont(QFont("Microsoft YaHei UI", 12, QFont::Bold));
        painter.drawText(panel.adjusted(14, 0, -14, -12),
                         Qt::AlignLeft | Qt::AlignBottom,
                         "状态：监听中");
    }

    painter.setBrush(QColor(34, 211, 238));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(QPoint(cx, cy), 4, 4);
}
