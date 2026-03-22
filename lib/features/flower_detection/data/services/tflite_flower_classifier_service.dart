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
  TensorType? _inputTensorType;
  TensorType? _outputTensorType;
  double _inputScale = 1.0;
  int _inputZeroPoint = 0;
  double _outputScale = 1.0;
  int _outputZeroPoint = 0;
  List<List<List<List<double>>>>? _inputTensor4D;
  List<List<List<List<int>>>>? _inputTensor4DQuantized;
  List<List<double>>? _outputTensor;
  Object? _dynamicOutputTensor;
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
    final Tensor inputTensor = _interpreter!.getInputTensor(0);
    final Tensor outputTensor = _interpreter!.getOutputTensor(0);

    _inputShape = inputTensor.shape;
    _outputShape = outputTensor.shape;
    _inputTensorType = inputTensor.type;
    _outputTensorType = outputTensor.type;
    _readQuantizationParams(inputTensor, isInput: true);
    _readQuantizationParams(outputTensor, isInput: false);
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

    final Object inputTensor = _prepareInputTensor(normalizedInput);
    final Object outputTensor = _outputTensorForRunOrThrow();

    interpreter.run(inputTensor, outputTensor);

    final List<double> scores = _extractScoresFromOutput(outputTensor);
    final int usableScoresCount =
        scores.length < _labels.length ? scores.length : _labels.length;

    if (usableScoresCount == 0) {
      throw StateError('Le modèle a retourné une sortie vide.');
    }

    final List<FlowerPrediction> predictions = List<FlowerPrediction>.generate(
      usableScoresCount,
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
    _inputTensorType = null;
    _outputTensorType = null;
    _inputScale = 1.0;
    _inputZeroPoint = 0;
    _outputScale = 1.0;
    _outputZeroPoint = 0;
    _inputTensor4D = null;
    _inputTensor4DQuantized = null;
    _outputTensor = null;
    _dynamicOutputTensor = null;
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

    final TensorType inputType = _inputTensorType ?? TensorType.float32;
    if (inputType == TensorType.float32) {
      _inputTensor4D = _createInputTensor(_inputShape);
      _inputTensor4DQuantized = null;
    } else if (inputType == TensorType.uint8 || inputType == TensorType.int8) {
      _inputTensor4DQuantized = _createQuantizedInputTensor(_inputShape);
      _inputTensor4D = null;
    } else {
      throw UnsupportedError('Type d\'entrée non supporté: $inputType');
    }

    final TensorType outputType = _outputTensorType ?? TensorType.float32;
    if (_outputShape.length == 2 && outputType == TensorType.float32) {
      _outputTensor = _createOutputTensor(_outputShape);
      _dynamicOutputTensor = null;
    } else if (outputType == TensorType.float32 ||
        outputType == TensorType.uint8 ||
        outputType == TensorType.int8) {
      _dynamicOutputTensor = _createDynamicTensor(
        _outputShape,
        outputType == TensorType.float32 ? 0.0 : 0,
      );
      _outputTensor = null;
    } else {
      throw UnsupportedError('Type de sortie non supporté: $outputType');
    }
  }

  List<List<List<List<double>>>> _inputTensorOrThrow() {
    final List<List<List<List<double>>>>? tensor = _inputTensor4D;
    if (tensor == null) {
      throw StateError(
          'Tenseur d\'entrée non initialisé. Appelle load() avant classify().');
    }
    return tensor;
  }

  Object _outputTensorForRunOrThrow() {
    final List<List<double>>? outputTensor2D = _outputTensor;
    if (outputTensor2D != null) {
      return outputTensor2D;
    }

    final Object? dynamicTensor = _dynamicOutputTensor;
    if (dynamicTensor == null) {
      throw StateError(
        'Tenseur de sortie non initialisé. Appelle load() avant classify().',
      );
    }

    return dynamicTensor;
  }

  Object _prepareInputTensor(Float32List normalizedInput) {
    final TensorType inputType = _inputTensorType ?? TensorType.float32;

    if (inputType == TensorType.float32) {
      final List<List<List<List<double>>>> inputTensor = _inputTensorOrThrow();
      _fillInputTensorFromFlat(normalizedInput, inputTensor);
      return inputTensor;
    }

    if (inputType == TensorType.uint8 || inputType == TensorType.int8) {
      final List<List<List<List<int>>>> inputTensor =
          _quantizedInputTensorOrThrow();
      _fillQuantizedInputTensorFromFlat(
        normalizedInput,
        inputTensor,
        inputType,
      );
      return inputTensor;
    }

    throw UnsupportedError('Type d\'entrée non supporté: $inputType');
  }

  List<double> _extractScoresFromOutput(Object outputTensor) {
    final List<num> flatValues = <num>[];
    _flattenNumericTensor(outputTensor, flatValues);

    if (flatValues.isEmpty) {
      return const <double>[];
    }

    final TensorType outputType = _outputTensorType ?? TensorType.float32;

    if (outputType == TensorType.float32) {
      return flatValues.map((num value) => value.toDouble()).toList();
    }

    if (outputType == TensorType.uint8) {
      return flatValues.map((num value) {
        final int quantized = value.toInt().clamp(0, 255);
        return (quantized - _outputZeroPoint) * _outputScale;
      }).toList();
    }

    if (outputType == TensorType.int8) {
      return flatValues.map((num value) {
        final int quantized = value.toInt().clamp(-128, 127);
        return (quantized - _outputZeroPoint) * _outputScale;
      }).toList();
    }

    throw UnsupportedError('Type de sortie non supporté: $outputType');
  }

  List<List<List<List<int>>>> _quantizedInputTensorOrThrow() {
    final List<List<List<List<int>>>>? tensor = _inputTensor4DQuantized;
    if (tensor == null) {
      throw StateError(
        'Tenseur d\'entrée quantifié non initialisé. Appelle load() avant classify().',
      );
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

  void _fillQuantizedInputTensorFromFlat(
    Float32List flatInput,
    List<List<List<List<int>>>> inputTensor,
    TensorType inputType,
  ) {
    int cursor = 0;
    for (final batch in inputTensor) {
      for (final row in batch) {
        for (final pixel in row) {
          for (int channelIndex = 0;
              channelIndex < pixel.length;
              channelIndex++) {
            final double value = flatInput[cursor++];
            final int quantized =
                ((value / _inputScale) + _inputZeroPoint).round();
            if (inputType == TensorType.uint8) {
              pixel[channelIndex] = quantized.clamp(0, 255);
            } else {
              pixel[channelIndex] = quantized.clamp(-128, 127);
            }
          }
        }
      }
    }
  }

  void _readQuantizationParams(Tensor tensor, {required bool isInput}) {
    final dynamic dynamicTensor = tensor;
    double scale = 1.0;
    int zeroPoint = 0;

    try {
      final dynamic params = dynamicTensor.params;
      final dynamic scaleValue = params?.scale;
      final dynamic zeroPointValue = params?.zeroPoint;

      if (scaleValue is num && scaleValue != 0) {
        scale = scaleValue.toDouble();
      }

      if (zeroPointValue is num) {
        zeroPoint = zeroPointValue.toInt();
      }
    } catch (_) {}

    if (isInput) {
      _inputScale = scale;
      _inputZeroPoint = zeroPoint;
      return;
    }

    _outputScale = scale;
    _outputZeroPoint = zeroPoint;
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

  List<List<List<List<int>>>> _createQuantizedInputTensor(List<int> shape) {
    final int batch = shape[0];
    final int height = shape[1];
    final int width = shape[2];
    final int channels = shape[3];

    return List<List<List<List<int>>>>.generate(batch, (_) {
      return List<List<List<int>>>.generate(height, (_) {
        return List<List<int>>.generate(width, (_) {
          return List<int>.filled(channels, 0);
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

  Object _createDynamicTensor(List<int> shape, num defaultValue) {
    if (shape.isEmpty) {
      return defaultValue;
    }

    if (shape.length == 1) {
      return List<num>.filled(shape.first, defaultValue, growable: false);
    }

    return List<Object>.generate(
      shape.first,
      (_) => _createDynamicTensor(shape.sublist(1), defaultValue),
      growable: false,
    );
  }

  void _flattenNumericTensor(Object value, List<num> out) {
    if (value is List) {
      for (final Object item in value) {
        _flattenNumericTensor(item, out);
      }
      return;
    }

    if (value is num) {
      out.add(value);
      return;
    }

    throw UnsupportedError(
        'Valeur non numérique dans la sortie modèle: $value');
  }
}
