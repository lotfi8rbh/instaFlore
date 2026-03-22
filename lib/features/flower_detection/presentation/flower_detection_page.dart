import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../../../core/widgets/app_loading_view.dart';
import '../data/services/camera_frame_preprocessor.dart';
import '../data/services/tflite_flower_classifier_service.dart';
import '../domain/entities/flower_prediction.dart';

class FlowerDetectionPage extends StatefulWidget {
  const FlowerDetectionPage({super.key});

  @override
  State<FlowerDetectionPage> createState() => _FlowerDetectionPageState();
}

class _FlowerDetectionPageState extends State<FlowerDetectionPage> {
  static const double _minimumReliableConfidence = 0.60;
  static const int _predictionSmoothingWindow = 5;

  final TfliteFlowerClassifierService _classifierService =
      TfliteFlowerClassifierService();

  CameraController? _cameraController;
  FlowerPrediction? _topPrediction;
  Float32List? _inputBuffer;

  bool _isInitializing = true;
  bool _isProcessingFrame = false;
  bool _isStreamPaused = false;
  String? _errorMessage;
  int _frameCounter = 0;
  final List<FlowerPrediction> _recentPredictions = <FlowerPrediction>[];

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _disposeCameraController();

    setState(() {
      _isInitializing = true;
      _errorMessage = null;
      _isProcessingFrame = false;
      _isStreamPaused = false;
      _topPrediction = null;
    });

    _inputBuffer = null;
    _frameCounter = 0;
    _recentPredictions.clear();

    try {
      final PermissionStatus permissionStatus =
          await Permission.camera.request();
      if (!permissionStatus.isGranted) {
        throw StateError('Permission caméra refusée.');
      }

      final List<CameraDescription> cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw StateError('Aucune caméra disponible sur cet appareil.');
      }

      final CameraDescription selectedCamera = cameras.firstWhere(
        (CameraDescription camera) =>
            camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final CameraController controller = CameraController(
        selectedCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isIOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.yuv420,
      );

      await controller.initialize();
      await _classifierService.load();

      _inputBuffer = CameraFramePreprocessor.allocateInputBuffer(
        targetWidth: _classifierService.inputWidth,
        targetHeight: _classifierService.inputHeight,
        targetChannels: _classifierService.inputChannels,
      );

      await controller.startImageStream(_onFrame);

      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _cameraController = controller;
      });
    } catch (error, stackTrace) {
      debugPrint('Erreur initialisation caméra/inférence: $error');
      debugPrintStack(stackTrace: stackTrace);

      if (!mounted) return;

      await _disposeCameraController();

      setState(() {
        _errorMessage = 'Initialisation échouée: $error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isInitializing = false;
        });
      }
    }
  }

  Future<void> _onFrame(CameraImage image) async {
    if (_isProcessingFrame ||
        !_classifierService.isReady ||
        _isStreamPaused ||
        _errorMessage != null) {
      return;
    }

    _frameCounter++;
    if (_frameCounter % 4 != 0) return;

    _isProcessingFrame = true;

    try {
      final Float32List? inputBuffer = _inputBuffer;
      if (inputBuffer == null) return;

      CameraFramePreprocessor.preprocessIntoBuffer(
        image,
        outputBuffer: inputBuffer,
        targetWidth: _classifierService.inputWidth,
        targetHeight: _classifierService.inputHeight,
        targetChannels: _classifierService.inputChannels,
        normalizationMode: InputNormalizationMode.mobileNetV2,
      );

      final List<FlowerPrediction> predictions = _classifierService.classify(
        inputBuffer,
        topK: 1,
      );

      if (!mounted || predictions.isEmpty) return;

      setState(() {
        _topPrediction = _getSmoothedPrediction(predictions.first);
      });
    } catch (error, stackTrace) {
      debugPrint('Erreur pendant l\'analyse temps réel: $error');
      debugPrintStack(stackTrace: stackTrace);

      if (!mounted) return;

      await _disposeCameraController();

      setState(() {
        _errorMessage = 'Erreur pendant l\'analyse temps réel: $error';
      });
    } finally {
      _isProcessingFrame = false;
    }
  }

  Future<void> _disposeCameraController() async {
    final CameraController? controller = _cameraController;
    _cameraController = null;

    if (controller == null) return;

    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (_) {}

    try {
      await controller.dispose();
    } catch (_) {}
  }

  void _retry() {
    if (_isInitializing) return;
    _initialize();
  }

  void _toggleStreamPause() {
    setState(() {
      _isStreamPaused = !_isStreamPaused;
    });
  }

  FlowerPrediction _getSmoothedPrediction(FlowerPrediction prediction) {
    _recentPredictions.add(prediction);
    if (_recentPredictions.length > _predictionSmoothingWindow) {
      _recentPredictions.removeAt(0);
    }

    final Map<String, List<double>> confidenceByLabel =
        <String, List<double>>{};
    for (final FlowerPrediction item in _recentPredictions) {
      confidenceByLabel.putIfAbsent(item.label, () => <double>[]).add(
            item.confidence,
          );
    }

    String selectedLabel = prediction.label;
    double selectedAverage = prediction.confidence;
    int selectedCount = -1;

    confidenceByLabel.forEach((String label, List<double> confidences) {
      final int count = confidences.length;
      final double average =
          confidences.reduce((double a, double b) => a + b) / count;

      if (count > selectedCount ||
          (count == selectedCount && average > selectedAverage)) {
        selectedLabel = label;
        selectedAverage = average;
        selectedCount = count;
      }
    });

    return FlowerPrediction(label: selectedLabel, confidence: selectedAverage);
  }

  @override
  void dispose() {
    _disposeCameraController();
    _classifierService.dispose();
    _inputBuffer = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final CameraController? controller = _cameraController;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: Colors.white,
        actions: <Widget>[
          if (!_isInitializing)
            IconButton(
              onPressed: _toggleStreamPause,
              tooltip: _isStreamPaused ? 'Reprendre' : 'Pause',
              icon: Icon(
                _isStreamPaused ? Icons.play_circle_outline : Icons.pause_circle_outline,
                size: 28,
              ),
            ),
        ],
      ),
      body: _buildBody(controller),
    );
  }

  Widget _buildBody(CameraController? controller) {
    if (_isInitializing) {
      return const AppLoadingView(message: 'Préparation de la caméra...');
    }

    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 20),
              FilledButton(
                onPressed: _retry,
                child: const Text('Réessayer'),
              ),
            ],
          ),
        ),
      );
    }

    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: Text('Caméra indisponible.'));
    }

    return Stack(
      fit: StackFit.expand,
      children: <Widget>[
        CameraPreview(controller),

        // Overlay résultat en bas
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            width: double.infinity,
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: <Color>[Colors.transparent, Colors.black87],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
            padding: const EdgeInsets.fromLTRB(24, 40, 24, 40),
            child: _buildResultOverlay(),
          ),
        ),

        // Badge "En pause"
        if (_isStreamPaused)
          Positioned(
            top: 100,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    Icon(Icons.pause, color: Colors.amber, size: 16),
                    SizedBox(width: 6),
                    Text(
                      'Analyse en pause',
                      style: TextStyle(color: Colors.amber, fontSize: 13),
                    ),
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildResultOverlay() {
    final FlowerPrediction? prediction = _topPrediction;

    if (prediction == null) {
      return const Text(
        'Pointez la caméra vers une fleur…',
        style: TextStyle(color: Colors.white70, fontSize: 16),
      );
    }

    final bool isReliable =
        prediction.confidence >= _minimumReliableConfidence;

    if (!isReliable) {
      return const Text(
        'Aucune fleur reconnue',
        style: TextStyle(color: Colors.white60, fontSize: 16),
      );
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          prediction.label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 32,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(height: 6),
        _ConfidenceBar(confidence: prediction.confidence),
      ],
    );
  }
}

class _ConfidenceBar extends StatelessWidget {
  const _ConfidenceBar({required this.confidence});

  final double confidence;

  @override
  Widget build(BuildContext context) {
    final int percent = (confidence * 100).round();
    final Color barColor = confidence >= 0.85
        ? Colors.greenAccent
        : confidence >= 0.70
            ? Colors.lightGreenAccent
            : Colors.orangeAccent;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          'Confiance $percent%',
          style: const TextStyle(color: Colors.white70, fontSize: 13),
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: confidence,
            backgroundColor: Colors.white24,
            valueColor: AlwaysStoppedAnimation<Color>(barColor),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}
