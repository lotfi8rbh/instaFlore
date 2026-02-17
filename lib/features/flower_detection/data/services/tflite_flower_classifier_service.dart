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
  List<List<List<List<double>>>>? _inputTensor4D;
  List<List<double>>? _outputTensor;
  int _flatInputLength = 0;

  bool get isReady => _interpreter != null && _labels.isNotEmpty;

  int get inputHeight => _inputShape.length >= 2 ? _inputShape[1] : 0;
  int get inputWidth => _inputShape.length >= 3 ? _inputShape[2] : 0;
  int get inputChannels => _inputShape.length >= 4 ? _inputShape[3] : 0;
  int get inputBufferLength => _flatInputLength;

  Future<void> load() async {
    if (_interpreter != null) {
      return;
    }

    _interpreter = await Interpreter.fromAsset(modelAssetPath);
    _inputShape = _interpreter!.getInputTensor(0).shape;
    _outputShape = _interpreter!.getOutputTensor(0).shape;
    _labels = await _loadLabels();
    _initializeReusableTensors();

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
    if (normalizedInput.length != _flatInputLength) {
      throw ArgumentError(
        'Taille du tenseur invalide. Reçu: ${normalizedInput.length}, attendu: $_flatInputLength',
      );
    }

    final List<List<List<List<double>>>> inputTensor = _inputTensorOrThrow();
    final List<List<double>> outputTensor = _outputTensorOrThrow();
    _fillInputTensorFromFlat(normalizedInput, inputTensor);

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
    _inputTensor4D = null;
    _outputTensor = null;
    _flatInputLength = 0;
  }

  Interpreter _interpreterOrThrow() {
    final Interpreter? interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError(
          'Le modèle n\'est pas chargé. Appelle load() avant classify().');
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

  void _initializeReusableTensors() {
    if (_inputShape.length != 4) {
      throw StateError('Shape d\'entrée non supporté: $_inputShape');
    }

    _flatInputLength = _inputShape.reduce(
      (int current, int next) => current * next,
    );

    _inputTensor4D = _createInputTensor(_inputShape);
    _outputTensor = _createOutputTensor(_outputShape);
  }

  List<List<List<List<double>>>> _inputTensorOrThrow() {
    final List<List<List<List<double>>>>? tensor = _inputTensor4D;
    if (tensor == null) {
      throw StateError(
          'Tenseur d\'entrée non initialisé. Appelle load() avant classify().');
    }
    return tensor;
  }

  List<List<double>> _outputTensorOrThrow() {
    final List<List<double>>? tensor = _outputTensor;
    if (tensor == null) {
      throw StateError(
          'Tenseur de sortie non initialisé. Appelle load() avant classify().');
    }
    return tensor;
  }

  void _fillInputTensorFromFlat(
    Float32List flatInput,
    List<List<List<List<double>>>> inputTensor,
  ) {
    int cursor = 0;
    for (final batch in inputTensor) {
      for (final row in batch) {
        for (final pixel in row) {
          for (int channelIndex = 0;
              channelIndex < pixel.length;
              channelIndex++) {
            pixel[channelIndex] = flatInput[cursor++];
          }
        }
      }
    }
  }

  List<List<List<List<double>>>> _createInputTensor(List<int> shape) {
    final int batch = shape[0];
    final int height = shape[1];
    final int width = shape[2];
    final int channels = shape[3];

    return List<List<List<List<double>>>>.generate(batch, (_) {
      return List<List<List<double>>>.generate(height, (_) {
        return List<List<double>>.generate(width, (_) {
          return List<double>.filled(channels, 0.0);
        });
      });
    });
  }

  List<List<double>> _createOutputTensor(List<int> shape) {
    final int batch = shape.isNotEmpty ? shape.first : 1;
    final int classesCount = shape.isNotEmpty ? shape.last : 0;

    return List<List<double>>.generate(
      batch,
      (_) => List<double>.filled(classesCount, 0.0),
    );
  }
}
