import 'package:flutter/material.dart';
import '../models/detection_result.dart';

/// Widget para display ang detection results with bounding boxes
///
/// Kini ang nagpakita sa boxes sa nakitang objects
class DetectionOverlay extends StatelessWidget {
  final List<DetectionResult> detections;
  final Size imageSize;

  const DetectionOverlay({
    super.key,
    required this.detections,
    required this.imageSize,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _DetectionPainter(detections, imageSize),
      child: Container(),
    );
  }
}

class _DetectionPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size imageSize;

  _DetectionPainter(this.detections, this.imageSize);

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width == 0 || imageSize.height == 0) {
      return;
    }

    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    for (var detection in detections) {
      final rect = detection.boundingBox;
      if (rect == null) {
        continue;
      }

      final box = Rect.fromLTRB(
        rect.left * scaleX,
        rect.top * scaleY,
        rect.right * scaleX,
        rect.bottom * scaleY,
      );

      // Draw outer shadow para mas klaro ang outline
      final shadowPaint = Paint()
        // ignore: deprecated_member_use
        ..color = Colors.black.withOpacity(0.4)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
      canvas.drawRect(box.inflate(4), shadowPaint);

      // Main bounding box
      final paint = Paint()
        ..color = Colors.green[400]!
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4.0;

      canvas.drawRect(box, paint);

      // Corner accents para mas dali makita
      final cornerPaint = Paint()
        ..color = Colors.green[300]!
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6.0
        ..strokeCap = StrokeCap.round;

      final cornerLength = 20.0;
      canvas.drawLine(
        Offset(box.left, box.top),
        Offset(box.left + cornerLength, box.top),
        cornerPaint,
      );
      canvas.drawLine(
        Offset(box.left, box.top),
        Offset(box.left, box.top + cornerLength),
        cornerPaint,
      );
      canvas.drawLine(
        Offset(box.right, box.top),
        Offset(box.right - cornerLength, box.top),
        cornerPaint,
      );
      canvas.drawLine(
        Offset(box.right, box.top),
        Offset(box.right, box.top + cornerLength),
        cornerPaint,
      );
      canvas.drawLine(
        Offset(box.left, box.bottom),
        Offset(box.left + cornerLength, box.bottom),
        cornerPaint,
      );
      canvas.drawLine(
        Offset(box.left, box.bottom),
        Offset(box.left, box.bottom - cornerLength),
        cornerPaint,
      );
      canvas.drawLine(
        Offset(box.right, box.bottom),
        Offset(box.right - cornerLength, box.bottom),
        cornerPaint,
      );
      canvas.drawLine(
        Offset(box.right, box.bottom),
        Offset(box.right, box.bottom - cornerLength),
        cornerPaint,
      );

      // Label text ug confidence
      final textSpan = TextSpan(
        text:
            '${detection.label}\n${(detection.confidence * 100).toStringAsFixed(1)}%',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.bold,
          height: 1.3,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
        textAlign: TextAlign.center,
      )..layout();

      final labelHeight = textPainter.height + 16;
      final labelWidth = textPainter.width + 24;

      final clampedLeft = box.left.clamp(0.0, size.width - labelWidth);
      final desiredTop = box.top - labelHeight - 8;
      final clampedTop = desiredTop.clamp(0.0, size.height - labelHeight);

      final labelRect = RRect.fromRectAndRadius(
        Rect.fromLTWH(clampedLeft, clampedTop, labelWidth, labelHeight),
        const Radius.circular(8),
      );

      final labelShadow = Paint()
        // ignore: deprecated_member_use
        ..color = Colors.black.withOpacity(0.3)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);
      canvas.drawRRect(labelRect.inflate(2), labelShadow);

      final labelBg = Paint()..color = Colors.green[700]!;
      canvas.drawRRect(labelRect, labelBg);

      textPainter.paint(canvas, Offset(clampedLeft + 12, clampedTop + 8));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
