import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' show Rect;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

import '../models/detection_result.dart';

/// Service para handle ang YOLOv8 model operations
///
/// Kini ang nag-manage sa model loading ug inference
class MLService {
  Interpreter? _interpreter;
  List<String>? _labels;
  Future<void>? _loadFuture; // Para malikayan ang multiple nga load attempts
  int? _numBoxes;
  int? _attrPerBox;
  bool _outputIsTransposed = false;

  // YOLOv8 640x640 input size
  static const int inputSize = 640;
  static const double confidenceThreshold =
      0.6; // Mas taas nga minimum confidence
  static const double iouThreshold = 0.5; // Non-max suppression

  /// Load ang TFLite model - gi-update para sa best_float32.tflite
  Future<void> loadModel() async {
    if (_interpreter != null) return;

    _loadFuture ??= _initializeModel();

    try {
      await _loadFuture;
      if (_interpreter == null) {
        _loadFuture = null;
        throw Exception(
          'Model attempt human pero walay interpreter. Tan-awa ang load logic.',
        );
      }
    } catch (e) {
      _loadFuture = null;
      rethrow;
    }
  }

  Future<void> _initializeModel() async {
    try {
      final interpreter = await Interpreter.fromAsset(
        'assets/models/bestN_float32.tflite',
      );
      final labels = await _loadLabels();

      _interpreter = interpreter;
      _labels = labels;
      _configureOutputMetadata(interpreter, labels.length);
    } catch (e) {
      _interpreter?.close();
      _interpreter = null;
      rethrow;
    }
  }

  /// Load ang class labels
  Future<List<String>> _loadLabels() async {
    try {
      final labelData = await rootBundle.loadString('assets/models/labels.txt');
      return labelData.split('\n').where((label) => label.isNotEmpty).toList();
    } catch (e) {
      return ['mangrove']; // Default label
    }
  }

  /// Run detection sa image
  Future<List<DetectionResult>> detectObjects(File imageFile) async {
    await loadModel();

    final interpreter = _interpreter;
    if (interpreter == null) {
      throw Exception(
        'Model wala pa gihapon na-load. Tan-awa ang logs para detalye.',
      );
    }

    try {
      // 1. Load ug preprocess ang image
      final image = img.decodeImage(await imageFile.readAsBytes());
      if (image == null) throw Exception('Cannot decode image');

      final inputImage = _preprocessImage(image);

      // 2. Prepare input tensor [1, 640, 640, 3]
      var input = inputImage.reshape([1, inputSize, inputSize, 3]);

      // 3. Prepare output tensor - YOLOv8 format for 15 classes
      // Gikuha nato ang aktwal nga shape gikan sa interpreter aron siguradong sakto
      final outputTensor = interpreter.getOutputTensor(0);
      final outputShape = outputTensor.shape;
      final totalElements = outputShape.fold<int>(
        1,
        (value, element) => value * element,
      );
      var output = List.filled(totalElements, 0.0).reshape(outputShape);

      // 4. Run inference
      interpreter.run(input, output);

      // 5. Process results
      final rawPredictions = output[0] as List;
      final formatted = _formatPredictions(rawPredictions);
      _debugOutputStatistics(formatted);

      final detections = _processYOLOv8Output(
        formatted,
        image.width,
        image.height,
      );

      return detections;
    } catch (e) {
      rethrow;
    }
  }

  /// Preprocess image para sa model input
  List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    // Resize to model input size
    final resized = img.copyResize(image, width: inputSize, height: inputSize);

    // Create 4D tensor [1, height, width, channels]
    var inputImage = List.generate(
      1,
      (_) => List.generate(
        inputSize,
        (y) => List.generate(inputSize, (x) {
          final pixel = resized.getPixel(x, y);
          return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
        }),
      ),
    );

    return inputImage;
  }

  /// Process ang YOLOv8 output to detection results
  List<DetectionResult> _processYOLOv8Output(
    List<List<double>> predictions,
    int imageWidth,
    int imageHeight,
  ) {
    List<DetectionResult> detections = [];

    if ((_attrPerBox ?? 0) <= 4) return detections;

    for (final prediction in predictions) {
      if (prediction.length < 4) continue;

      // GI-normalize nato ang class scores gamit sigmo aron mo-range sa 0-1
      final centerX = prediction[0];
      final centerY = prediction[1];
      final width = prediction[2];
      final height = prediction[3];

      if (width <= 0 || height <= 0) {
        continue;
      }

      double maxClassScore = 0.0;
      int classId = 0;

      for (int c = 0; c < prediction.length - 4; c++) {
        final rawScore = prediction[4 + c];
        final classScore = _sigmoid(rawScore);
        if (classScore > maxClassScore) {
          maxClassScore = classScore;
          classId = c;
        }
      }

      final confidence = maxClassScore;

      if (confidence > confidenceThreshold) {
        final left = (centerX - width / 2) * imageWidth / inputSize;
        final top = (centerY - height / 2) * imageHeight / inputSize;
        final right = (centerX + width / 2) * imageWidth / inputSize;
        final bottom = (centerY + height / 2) * imageHeight / inputSize;

        final label = _labels != null && classId < _labels!.length
            ? _labels![classId]
            : 'Unknown';

        detections.add(
          DetectionResult(
            label: label,
            confidence: confidence,
            boundingBox: Rect.fromLTRB(
              left.clamp(0, imageWidth.toDouble()),
              top.clamp(0, imageHeight.toDouble()),
              right.clamp(0, imageWidth.toDouble()),
              bottom.clamp(0, imageHeight.toDouble()),
            ),
          ),
        );
      }
    }

    // Apply non-maximum suppression
    return _nonMaxSuppression(detections);
  }

  /// Non-maximum suppression para remove duplicate detections
  List<DetectionResult> _nonMaxSuppression(List<DetectionResult> detections) {
    // Sort by confidence
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<DetectionResult> results = [];

    for (var detection in detections) {
      bool keep = true;

      for (var kept in results) {
        if (_calculateIoU(detection.boundingBox, kept.boundingBox) >
            iouThreshold) {
          keep = false;
          break;
        }
      }

      if (keep) results.add(detection);
    }

    return results;
  }

  /// Calculate Intersection over Union
  double _calculateIoU(Rect box1, Rect box2) {
    final intersection = box1.intersect(box2);
    if (intersection.width <= 0 || intersection.height <= 0) return 0.0;

    final intersectionArea = intersection.width * intersection.height;
    final union =
        box1.width * box1.height + box2.width * box2.height - intersectionArea;

    return intersectionArea / union;
  }

  /// Dispose ang interpreter
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loadFuture = null;
  }

  /// Configure ang metadata sa output aron sakto nato ma-interpret ang tensor
  void _configureOutputMetadata(Interpreter interpreter, int labelCount) {
    final outputTensor = interpreter.getOutputTensor(0);
    final shape = outputTensor.shape;

    if (shape.length != 3) {
      throw StateError(
        'Gi-expect nga 3D ang output tensor pero ${shape.length}D ang nadawat.',
      );
    }

    final expectedAttr = labelCount + 4;
    final dim1 = shape[1];
    final dim2 = shape[2];

    if (dim2 == expectedAttr) {
      _attrPerBox = dim2;
      _numBoxes = dim1;
      _outputIsTransposed = false;
    } else if (dim1 == expectedAttr) {
      _attrPerBox = dim1;
      _numBoxes = dim2;
      _outputIsTransposed = true;
    } else if (dim1 > dim2) {
      _attrPerBox = dim2;
      _numBoxes = dim1;
      _outputIsTransposed = false;
    } else {
      _attrPerBox = dim1;
      _numBoxes = dim2;
      _outputIsTransposed = true;
    }

    if (_attrPerBox == null || _attrPerBox! < 5) {
      throw StateError('Kulangan ang attributes per box: $_attrPerBox');
    }
  }

  /// I-format nato ang raw output ngadto sa lista sa mga kahon nga dali i-parse
  List<List<double>> _formatPredictions(List raw) {
    final numBoxes = _numBoxes;
    final attrPerBox = _attrPerBox;

    if (numBoxes == null || attrPerBox == null) {
      throw StateError('Wala ma-configure ang output metadata sa model.');
    }

    if (!_outputIsTransposed) {
      if (raw.length != numBoxes) {
        throw StateError(
          'Gi-expect $numBoxes ka kahon pero ${raw.length} ang nakuha.',
        );
      }

      return raw
          .map<List<double>>(
            (box) => (box as List)
                .map((value) => (value as num).toDouble())
                .toList(),
          )
          .toList();
    }

    if (raw.length != attrPerBox) {
      throw StateError(
        'Gi-expect $attrPerBox ka attributes pero ${raw.length} ang nakuha.',
      );
    }

    return List.generate(numBoxes, (boxIdx) {
      return List.generate(attrPerBox, (attrIdx) {
        final attrList = raw[attrIdx] as List;
        return (attrList[boxIdx] as num).toDouble();
      });
    });
  }

  /// Sigmoid para sa pag-normalize sa scores ngadto 0-1 range
  double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  /// Debug helper para makit-an nato ang sample scores sa raw predictions
  void _debugOutputStatistics(List<List<double>> predictions) {
    if (!kDebugMode || predictions.isEmpty) return;

    double globalMax = 0;
    double globalMin = double.infinity;
    double bestScore = 0;
    int bestIndex = -1;

    for (int i = 0; i < math.min(50, predictions.length); i++) {
      final box = predictions[i];
      if (box.length < 5) continue;

      globalMin = math.min(globalMin, box.reduce(math.min));
      globalMax = math.max(globalMax, box.reduce(math.max));

      for (int c = 4; c < box.length; c++) {
        final score = _sigmoid(box[c]);
        if (score > bestScore) {
          bestScore = score;
          bestIndex = i;
        }
      }
    }

    if (globalMin == double.infinity) globalMin = 0;
    debugPrint(
      'MLService debug -> raw min: ${globalMin.toStringAsFixed(4)}, raw max: ${globalMax.toStringAsFixed(4)}, best score: ${bestScore.toStringAsFixed(4)} @ box $bestIndex',
    );

    if (bestIndex >= 0) {
      final sample = predictions[bestIndex];
      debugPrint(
        'Sample box[$bestIndex]: x=${sample[0].toStringAsFixed(2)}, y=${sample[1].toStringAsFixed(2)}, w=${sample[2].toStringAsFixed(2)}, h=${sample[3].toStringAsFixed(2)}',
      );
    }
  }
}
