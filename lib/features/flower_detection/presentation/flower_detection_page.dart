import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../data/services/camera_frame_preprocessor.dart';
import '../data/services/tflite_flower_classifier_service.dart';
import '../domain/entities/flower_prediction.dart';

class FlowerDetectionPage extends StatefulWidget {
  const FlowerDetectionPage({super.key});

  @override
  State<FlowerDetectionPage> createState() => _FlowerDetectionPageState();
}

class _FlowerDetectionPageState extends State<FlowerDetectionPage> {
  final TfliteFlowerClassifierService _classifierService =
      TfliteFlowerClassifierService();

  CameraController? _cameraController;
  FlowerPrediction? _topPrediction;
  Float32List? _inputBuffer;

  bool _isInitializing = true;
  bool _isProcessingFrame = false;
  String? _errorMessage;
  double? _lastInferenceMs;
  int _frameCounter = 0;

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
      _topPrediction = null;
      _lastInferenceMs = null;
    });

    _inputBuffer = null;
    _frameCounter = 0;

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

      if (!mounted) {
        return;
      }

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
        _errorMessage != null) {
      return;
    }

    _frameCounter++;
    if (_frameCounter % 4 != 0) {
      return;
    }

    _isProcessingFrame = true;
    final Stopwatch stopwatch = Stopwatch()..start();

    try {
      final Float32List? inputBuffer = _inputBuffer;
      if (inputBuffer == null) {
        return;
      }

      CameraFramePreprocessor.preprocessIntoBuffer(
        image,
        outputBuffer: inputBuffer,
        targetWidth: _classifierService.inputWidth,
        targetHeight: _classifierService.inputHeight,
        targetChannels: _classifierService.inputChannels,
      );

      final List<FlowerPrediction> predictions = _classifierService.classify(
        inputBuffer,
        topK: 1,
      );

      if (!mounted || predictions.isEmpty) {
        return;
      }

      stopwatch.stop();
      setState(() {
        _topPrediction = predictions.first;
        _lastInferenceMs = stopwatch.elapsedMicroseconds / 1000;
      });
    } catch (error, stackTrace) {
      debugPrint('Erreur pendant l\'analyse temps réel: $error');
      debugPrintStack(stackTrace: stackTrace);

      if (!mounted) {
        return;
      }

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

    if (controller == null) {
      return;
    }

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
    if (_isInitializing) {
      return;
    }
    _initialize();
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
      appBar: AppBar(title: const Text('InstaFlore - Détection temps réel')),
      body: _buildBody(controller),
    );
  }

  Widget _buildBody(CameraController? controller) {
    if (_isInitializing) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              ElevatedButton(
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
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            width: double.infinity,
            color: Colors.black54,
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  _topPrediction == null
                      ? 'Analyse en cours...'
                      : 'Fleur: ${_topPrediction!.label}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  _topPrediction == null
                      ? 'Confiance: --'
                      : 'Confiance: ${(_topPrediction!.confidence * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(color: Colors.white),
                ),
                const SizedBox(height: 4),
                Text(
                  _lastInferenceMs == null
                      ? 'Latence: --'
                      : 'Latence: ${_lastInferenceMs!.toStringAsFixed(1)} ms',
                  style: const TextStyle(color: Colors.white70),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
