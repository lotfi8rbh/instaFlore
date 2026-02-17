import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../../domain/entities/flower_prediction.dart';

class TfliteFlowerClassifierService {
  static const String modelAssetPath = 'assets/model_clustered.tflite';
  static const String labelsAssetPath = 'assets/labels.txt';

  Interpreter? _interpreter;
  List<String> _labels = const [];
  List<int> _inputShape = const [];
  List<int> _outputShape = const [];

  bool get isReady => _interpreter != null && _labels.isNotEmpty;

  int get inputHeight => _inputShape.length >= 2 ? _inputShape[1] : 0;
  int get inputWidth => _inputShape.length >= 3 ? _inputShape[2] : 0;
  int get inputChannels => _inputShape.length >= 4 ? _inputShape[3] : 0;

  Future<void> load() async {
    if (_interpreter != null) {
      return;
    }

    _interpreter = await Interpreter.fromAsset(modelAssetPath);
    _inputShape = _interpreter!.getInputTensor(0).shape;
    _outputShape = _interpreter!.getOutputTensor(0).shape;
    _labels = await _loadLabels();

    final int classesCount = _outputShape.last;
    if (_labels.length != classesCount) {
      throw StateError(
        'Le nombre de labels (${_labels.length}) ne correspond pas au nombre de sorties du modèle ($classesCount).',
      );
    }
  }

  List<FlowerPrediction> classify(
    Float32List normalizedInput, {
    int topK = 1,
  }) {
    final Interpreter interpreter = _interpreterOrThrow();
    if (_inputShape.length != 4) {
      throw StateError('Shape d\'entrée non supporté: $_inputShape');
    }

    final int expectedInputLength = _inputShape.reduce(
      (int current, int next) => current * next,
    );

    if (normalizedInput.length != expectedInputLength) {
      throw ArgumentError(
        'Taille du tenseur invalide. Reçu: ${normalizedInput.length}, attendu: $expectedInputLength',
      );
    }

    final Object inputTensor = _reshapeTo4D(normalizedInput, _inputShape);
    final int classesCount = _outputShape.last;
    final List<List<double>> outputTensor = List<List<double>>.generate(
      1,
      (_) => List<double>.filled(classesCount, 0.0),
    );

    interpreter.run(inputTensor, outputTensor);

    final List<double> scores = outputTensor.first;
    final List<FlowerPrediction> predictions = List<FlowerPrediction>.generate(
      scores.length,
      (int index) => FlowerPrediction(
        label: _labels[index],
        confidence: scores[index],
      ),
    )..sort(
        (FlowerPrediction a, FlowerPrediction b) =>
            b.confidence.compareTo(a.confidence),
      );

    final int safeTopK = topK.clamp(1, predictions.length);
    return predictions.take(safeTopK).toList(growable: false);
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _labels = const [];
    _inputShape = const [];
    _outputShape = const [];
  }

  Interpreter _interpreterOrThrow() {
    final Interpreter? interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('Le modèle n\'est pas chargé. Appelle load() avant classify().');
    }
    return interpreter;
  }

  Future<List<String>> _loadLabels() async {
    final String rawLabels = await rootBundle.loadString(labelsAssetPath);
    final List<String> labels = rawLabels
        .split('\n')
        .map((String line) => line.trim())
        .where((String line) => line.isNotEmpty)
        .toList(growable: false);

    if (labels.isEmpty) {
      throw StateError('Aucun label trouvé dans $labelsAssetPath');
    }

    return labels;
  }

  List<List<List<List<double>>>> _reshapeTo4D(
    Float32List flatInput,
    List<int> shape,
  ) {
    final int batch = shape[0];
    final int height = shape[1];
    final int width = shape[2];
    final int channels = shape[3];

    int cursor = 0;

    return List<List<List<List<double>>>>.generate(batch, (_) {
      return List<List<List<double>>>.generate(height, (_) {
        return List<List<double>>.generate(width, (_) {
          return List<double>.generate(channels, (_) {
            return flatInput[cursor++];
          });
        });
      });
    });
  }
}
