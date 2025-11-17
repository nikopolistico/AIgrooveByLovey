import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection_result.dart';

/// Service para mo-handle sa image classification gamit TFLite
///
/// Nag-manage ni sa model loading, preprocessing, ug pagbalik sa top prediction
class MLService {
  Interpreter? _interpreter;
  List<String>? _labels;
  Future<void>? _loadFuture; // Aron dili mag double load attempts

  int? _inputWidth;
  int? _inputHeight;
  int? _inputChannels;

  static const double confidenceThreshold =
      0.9; // Minimum 90% nga kasigurohan nga atong dawaton

  int? get inputWidth => _inputWidth;
  int? get inputHeight => _inputHeight;

  /// Gigamit sa UI aron kabalo sa target nga gidak-on kung mag-resize ta sa hulagway
  int get inputImageSize => _inputWidth ?? _inputHeight ?? 224;

  /// Public entry para i-load ang classification model
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
        'assets/models/best_float32.tflite',
      );
      final labels = await _loadLabels();

      _configureInputMetadata(interpreter);

      _interpreter = interpreter;
      _labels = labels;
    } catch (e) {
      _interpreter?.close();
      _interpreter = null;
      rethrow;
    }
  }

  /// Kuha ug parse sa labels gikan sa assets
  Future<List<String>> _loadLabels() async {
    try {
      final labelData = await rootBundle.loadString('assets/models/labels.txt');
      return labelData.split('\n').where((label) => label.isNotEmpty).toList();
    } catch (e) {
      return ['mangrove']; // Default fallback kung walay labels
    }
  }

  /// Main method nga mo-classify sa hulagway
  Future<List<DetectionResult>> detectObjects(File imageFile) async {
    await loadModel();

    final interpreter = _interpreter;
    if (interpreter == null) {
      throw Exception(
        'Model wala pa gihapon na-load. Tan-awa ang logs para detalye.',
      );
    }

    try {
      final image = img.decodeImage(await imageFile.readAsBytes());
      if (image == null) throw Exception('Cannot decode image');

      final inputTensor = _preprocessImage(image);
      final inputShape = interpreter.getInputTensor(0).shape;
      final reshapedInput = inputTensor.reshape(inputShape);

      final outputTensor = interpreter.getOutputTensor(0);
      final outputShape = outputTensor.shape;
      final totalElements = outputShape.fold<int>(1, (v, e) => v * e);
      final outputBuffer = List.filled(totalElements, 0.0).reshape(outputShape);

      interpreter.run(reshapedInput, outputBuffer);

      final rawScores = _extractScores(outputBuffer);
      final probabilities = _normalizeScores(rawScores);

      final bestIndex = _argMax(probabilities);
      if (bestIndex == null) return [];

      final bestScore = probabilities[bestIndex];
      if (bestScore < confidenceThreshold) return [];

      final label = (_labels != null && bestIndex < _labels!.length)
          ? _labels![bestIndex]
          : 'Unknown';

      _debugTopK(probabilities);

      return [DetectionResult(label: label, confidence: bestScore)];
    } catch (e) {
      rethrow;
    }
  }

  /// Gamiton nato aron mahibaloan ang eksaktong input shape sa model
  void _configureInputMetadata(Interpreter interpreter) {
    final inputTensor = interpreter.getInputTensor(0);
    final shape = inputTensor.shape;

    if (shape.length != 4) {
      throw StateError(
        'Gi-expect nga 4D ang input tensor (1, H, W, C) pero ${shape.length}D ang nadawat.',
      );
    }

    _inputHeight = shape[1];
    _inputWidth = shape[2];
    _inputChannels = shape[3];

    if (_inputChannels != 3) {
      throw StateError(
        'Gi-expect 3 channels (RGB) pero nakuha nato: $_inputChannels',
      );
    }
  }

  /// Preprocess nga mo-resize ug mo-normalize sa hulagway para sa model
  List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    final targetWidth = _inputWidth;
    final targetHeight = _inputHeight;

    if (targetWidth == null || targetHeight == null) {
      throw StateError('Wala ma-configure ang input tensor metadata.');
    }

    if (targetWidth <= 0 || targetHeight <= 0) {
      throw StateError('Invalid target size gikan sa model metadata.');
    }

    final resized = img.copyResize(
      image,
      width: targetWidth,
      height: targetHeight,
    );

    return List.generate(
      1,
      (_) => List.generate(
        targetHeight,
        (y) => List.generate(targetWidth, (x) {
          final pixel = resized.getPixel(x, y);
          return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
        }),
      ),
    );
  }

  // Flatten ug himoon nga List<double> ang output buffer gikan sa interpreter
  List<double> _extractScores(dynamic tensorOutput) {
    if (tensorOutput is List) {
      if (tensorOutput.isEmpty) return List<double>.empty();
      if (tensorOutput.first is List) {
        return _extractScores(tensorOutput.first);
      }
      return tensorOutput
          .map<double>((value) => (value as num).toDouble())
          .toList();
    }
    return List<double>.empty();
  }

  /// Normalize ang raw scores ug apply softmax kung kinahanglan
  List<double> _normalizeScores(List<double> rawScores) {
    if (rawScores.isEmpty) return rawScores;

    final hasNegative = rawScores.any((value) => value < 0);
    final hasAboveOne = rawScores.any((value) => value > 1);

    if (hasNegative || hasAboveOne) {
      final maxLogit = rawScores.reduce(math.max);
      final expValues = rawScores
          .map((value) => math.exp(value - maxLogit))
          .toList(growable: false);
      final sumExp = expValues.fold<double>(0, (sum, value) => sum + value);
      if (sumExp == 0) {
        return List<double>.filled(rawScores.length, 0);
      }
      return expValues.map((value) => value / sumExp).toList(growable: false);
    }

    final sum = rawScores.fold<double>(0, (sum, value) => sum + value);
    if (sum == 0) return rawScores;
    return rawScores.map((value) => value / sum).toList(growable: false);
  }

  /// Kuha sa index sa pinaka taas nga probability
  int? _argMax(List<double> values) {
    if (values.isEmpty) return null;

    var bestIndex = 0;
    var bestScore = values.first;

    for (var i = 1; i < values.length; i++) {
      if (values[i] > bestScore) {
        bestScore = values[i];
        bestIndex = i;
      }
    }

    return bestIndex;
  }

  /// Debug helper aron makita ang top predictions kung debug mode
  void _debugTopK(List<double> probabilities) {
    if (!kDebugMode || probabilities.isEmpty || _labels == null) return;

    final entries = List.generate(probabilities.length, (index) {
      return MapEntry(index, probabilities[index]);
    });

    entries.sort((a, b) => b.value.compareTo(a.value));
    final top = entries
        .take(3)
        .map((entry) {
          final label = entry.key < _labels!.length
              ? _labels![entry.key]
              : 'Unknown-${entry.key}';
          return '$label: ${(entry.value * 100).toStringAsFixed(1)}%';
        })
        .join(', ');

    debugPrint('MLService debug -> top predictions: $top');
  }

  /// Release resources kung dili na gamiton ang model
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loadFuture = null;
  }
}
