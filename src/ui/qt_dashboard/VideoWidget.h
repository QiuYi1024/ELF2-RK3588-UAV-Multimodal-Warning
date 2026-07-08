#pragma once
#include <QWidget>
#include <QtGlobal>

class QTimer;

class VideoWidget : public QWidget {
    Q_OBJECT
public:
    explicit VideoWidget(QWidget* parent = nullptr);

signals:
    void videoPointClicked(double normalizedX, double normalizedY);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QTimer* refreshTimer = nullptr;
    QRect lastVideoRect;
    int lastPaintedFrameId = -2;
    qint64 lastOverlayUpdateMonoMs = 0;
};
