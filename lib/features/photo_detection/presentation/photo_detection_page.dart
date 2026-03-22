import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../../core/widgets/app_loading_view.dart';
import '../../flower_detection/data/services/tflite_flower_classifier_service.dart';
import '../../flower_detection/domain/entities/flower_prediction.dart';

class PhotoDetectionPage extends StatefulWidget {
  const PhotoDetectionPage({super.key});

  @override
  State<PhotoDetectionPage> createState() => _PhotoDetectionPageState();
}

class _PhotoDetectionPageState extends State<PhotoDetectionPage> {
  static const double _minimumReliableConfidence = 0.60;

  final TfliteFlowerClassifierService _classifierService =
      TfliteFlowerClassifierService();
  final ImagePicker _imagePicker = ImagePicker();

  XFile? _pickedFile;
  FlowerPrediction? _prediction;
  bool _isLoading = false;
  bool _isModelReady = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      await _classifierService.load();
      if (mounted) {
        setState(() {
          _isModelReady = true;
        });
      }
    } catch (error) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Impossible de charger le modèle: $error';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _pickAndAnalyze() async {
    final XFile? file = await _imagePicker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1024,
      maxHeight: 1024,
    );

    if (file == null || !mounted) return;

    setState(() {
      _pickedFile = file;
      _prediction = null;
      _errorMessage = null;
      _isLoading = true;
    });

    try {
      final Uint8List imageBytes = await file.readAsBytes();
      final ui.Codec codec = await ui.instantiateImageCodec(imageBytes);
      final ui.FrameInfo frame = await codec.getNextFrame();
      final ui.Image image = frame.image;
      final int sourceWidth = image.width;
      final int sourceHeight = image.height;

      final ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      image.dispose();

      if (byteData == null) {
        throw StateError('Impossible de décoder les pixels de l\'image.');
      }

      final List<FlowerPrediction> predictions =
          _classifierService.classifyRgbaImage(
        byteData.buffer.asUint8List(),
        sourceWidth: sourceWidth,
        sourceHeight: sourceHeight,
        topK: 1,
      );

      if (!mounted) return;

      setState(() {
        _prediction = predictions.isNotEmpty ? predictions.first : null;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Erreur lors de l\'analyse: $error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _classifierService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Analyser une photo')),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isLoading && !_isModelReady) {
      return const AppLoadingView(message: 'Préparation du modèle...');
    }

    if (_errorMessage != null && !_isModelReady) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 16),
              Text(_errorMessage!, textAlign: TextAlign.center),
              const SizedBox(height: 20),
              FilledButton(
                onPressed: _loadModel,
                child: const Text('Réessayer'),
              ),
            ],
          ),
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          _buildImageArea(),
          const SizedBox(height: 20),
          FilledButton.icon(
            onPressed: _isLoading ? null : _pickAndAnalyze,
            icon: const Icon(Icons.photo_library_outlined),
            label: Text(
              _pickedFile == null ? 'Choisir une photo' : 'Changer de photo',
            ),
          ),
          if (_isLoading && _isModelReady) ...<Widget>[
            const SizedBox(height: 32),
            const Center(child: CircularProgressIndicator()),
            const SizedBox(height: 12),
            const Center(
              child: Text(
                'Analyse en cours…',
                style: TextStyle(fontSize: 15),
              ),
            ),
          ],
          if (_errorMessage != null && _isModelReady) ...<Widget>[
            const SizedBox(height: 16),
            _buildErrorCard(_errorMessage!),
          ],
          if (_prediction != null && !_isLoading) ...<Widget>[
            const SizedBox(height: 24),
            _buildResultCard(_prediction!),
          ],
        ],
      ),
    );
  }

  Widget _buildImageArea() {
    final XFile? file = _pickedFile;

    if (file == null) {
      return GestureDetector(
        onTap: _isLoading ? null : _pickAndAnalyze,
        child: Container(
          height: 280,
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: Theme.of(context).colorScheme.outlineVariant,
              width: 1.5,
            ),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Icon(
                Icons.add_photo_alternate_outlined,
                size: 64,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 12),
              Text(
                'Appuyez pour choisir une photo',
                style: TextStyle(
                  fontSize: 15,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Image.file(
        File(file.path),
        height: 280,
        width: double.infinity,
        fit: BoxFit.cover,
      ),
    );
  }

  Widget _buildResultCard(FlowerPrediction prediction) {
    final bool isReliable =
        prediction.confidence >= _minimumReliableConfidence;
    final ColorScheme colors = Theme.of(context).colorScheme;

    if (!isReliable) {
      return Card(
        color: colors.surfaceContainerHighest,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: <Widget>[
              Icon(Icons.help_outline, color: colors.onSurfaceVariant),
              const SizedBox(width: 12),
              Text(
                'Fleur non identifiée',
                style: TextStyle(
                  fontSize: 16,
                  color: colors.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      );
    }

    final int percent = (prediction.confidence * 100).round();

    return Card(
      color: colors.primaryContainer,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: <Widget>[
                Icon(Icons.check_circle_outline,
                    color: colors.onPrimaryContainer),
                const SizedBox(width: 8),
                Text(
                  'Fleur identifiée',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: colors.onPrimaryContainer,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              prediction.label,
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: colors.onPrimaryContainer,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              'Confiance $percent%',
              style: TextStyle(
                fontSize: 13,
                color: colors.onPrimaryContainer.withOpacity(0.75),
              ),
            ),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: prediction.confidence,
                backgroundColor: colors.onPrimaryContainer.withOpacity(0.15),
                valueColor: AlwaysStoppedAnimation<Color>(
                  colors.onPrimaryContainer,
                ),
                minHeight: 6,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorCard(String message) {
    return Card(
      color: Theme.of(context).colorScheme.errorContainer,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(
          message,
          style: TextStyle(
            color: Theme.of(context).colorScheme.onErrorContainer,
          ),
        ),
      ),
    );
  }
}
