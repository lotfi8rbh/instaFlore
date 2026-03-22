import 'dart:typed_data';

import 'package:camera/camera.dart';

enum InputNormalizationMode {
  zeroToOne,
  mobileNetV2,
}

class CameraFramePreprocessor {
  const CameraFramePreprocessor._();

  static Float32List allocateInputBuffer({
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
  }) {
    _validateTargetDimensions(
      targetWidth: targetWidth,
      targetHeight: targetHeight,
      targetChannels: targetChannels,
    );

    return Float32List(targetWidth * targetHeight * targetChannels);
  }

  static Float32List preprocess(
    CameraImage image, {
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
    InputNormalizationMode normalizationMode = InputNormalizationMode.zeroToOne,
  }) {
    final Float32List outputBuffer = allocateInputBuffer(
      targetWidth: targetWidth,
      targetHeight: targetHeight,
      targetChannels: targetChannels,
    );

    preprocessIntoBuffer(
      image,
      outputBuffer: outputBuffer,
      targetWidth: targetWidth,
      targetHeight: targetHeight,
      targetChannels: targetChannels,
      normalizationMode: normalizationMode,
    );

    return outputBuffer;
  }

  static void preprocessIntoBuffer(
    CameraImage image, {
    required Float32List outputBuffer,
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
    InputNormalizationMode normalizationMode = InputNormalizationMode.zeroToOne,
  }) {
    _validateTargetDimensions(
      targetWidth: targetWidth,
      targetHeight: targetHeight,
      targetChannels: targetChannels,
    );

    final int expectedLength = targetWidth * targetHeight * targetChannels;
    if (outputBuffer.length != expectedLength) {
      throw ArgumentError(
        'Longueur du buffer invalide. Reçu: ${outputBuffer.length}, attendu: $expectedLength',
      );
    }

    if (image.format.group == ImageFormatGroup.yuv420) {
      _preprocessYuv420(
        image,
        outputBuffer: outputBuffer,
        targetWidth: targetWidth,
        targetHeight: targetHeight,
        targetChannels: targetChannels,
        normalizationMode: normalizationMode,
      );
      return;
    }

    if (image.format.group == ImageFormatGroup.bgra8888) {
      _preprocessBgra8888(
        image,
        outputBuffer: outputBuffer,
        targetWidth: targetWidth,
        targetHeight: targetHeight,
        targetChannels: targetChannels,
        normalizationMode: normalizationMode,
      );
      return;
    }

    throw UnsupportedError(
      'Format caméra non supporté: ${image.format.group}',
    );
  }

  static void _preprocessYuv420(
    CameraImage image, {
    required Float32List outputBuffer,
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
    required InputNormalizationMode normalizationMode,
  }) {
    final Plane yPlane = image.planes.first;

    int cursor = 0;

    for (int y = 0; y < targetHeight; y++) {
      final int sourceY = (y * image.height / targetHeight).floor();
      final int sourceYClamped = sourceY.clamp(0, image.height - 1);

      for (int x = 0; x < targetWidth; x++) {
        final int sourceX = (x * image.width / targetWidth).floor();
        final int sourceXClamped = sourceX.clamp(0, image.width - 1);

        final int yIndex =
            (sourceYClamped * yPlane.bytesPerRow) + sourceXClamped;
        final int luma = yPlane.bytes[yIndex];
        final double normalized = _normalize(
          luma / 255.0,
          normalizationMode,
        );

        for (int channel = 0; channel < targetChannels; channel++) {
          outputBuffer[cursor++] = normalized;
        }
      }
    }
  }

  static void _preprocessBgra8888(
    CameraImage image, {
    required Float32List outputBuffer,
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
    required InputNormalizationMode normalizationMode,
  }) {
    final Plane plane = image.planes.first;
    final int bytesPerPixel = plane.bytesPerPixel ?? 4;

    int cursor = 0;

    for (int y = 0; y < targetHeight; y++) {
      final int sourceY = (y * image.height / targetHeight).floor();
      final int sourceYClamped = sourceY.clamp(0, image.height - 1);

      for (int x = 0; x < targetWidth; x++) {
        final int sourceX = (x * image.width / targetWidth).floor();
        final int sourceXClamped = sourceX.clamp(0, image.width - 1);

        final int pixelIndex = (sourceYClamped * plane.bytesPerRow) +
            (sourceXClamped * bytesPerPixel);

        final int blue = plane.bytes[pixelIndex];
        final int green = plane.bytes[pixelIndex + 1];
        final int red = plane.bytes[pixelIndex + 2];

        if (targetChannels == 1) {
          final double luminance =
              ((0.299 * red) + (0.587 * green) + (0.114 * blue)) / 255.0;
          outputBuffer[cursor++] = _normalize(luminance, normalizationMode);
        } else {
          outputBuffer[cursor++] = _normalize(red / 255.0, normalizationMode);
          outputBuffer[cursor++] = _normalize(green / 255.0, normalizationMode);
          outputBuffer[cursor++] = _normalize(blue / 255.0, normalizationMode);

          for (int channel = 3; channel < targetChannels; channel++) {
            outputBuffer[cursor++] = 0;
          }
        }
      }
    }
  }

  /// Prétraite des bytes RGBA bruts (ex: issus de dart:ui Image.toByteData)
  /// vers le buffer d'entrée du modèle.
  ///
  /// [rgbaBytes] : pixels en format RGBA, 4 octets par pixel, ligne par ligne.
  /// [sourceWidth] / [sourceHeight] : dimensions de l'image source.
  static void preprocessRgbaIntoBuffer(
    Uint8List rgbaBytes, {
    required int sourceWidth,
    required int sourceHeight,
    required Float32List outputBuffer,
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
    InputNormalizationMode normalizationMode = InputNormalizationMode.zeroToOne,
  }) {
    _validateTargetDimensions(
      targetWidth: targetWidth,
      targetHeight: targetHeight,
      targetChannels: targetChannels,
    );

    final int expectedLength = targetWidth * targetHeight * targetChannels;
    if (outputBuffer.length != expectedLength) {
      throw ArgumentError(
        'Longueur du buffer invalide. Reçu: ${outputBuffer.length}, attendu: $expectedLength',
      );
    }

    if (rgbaBytes.length != sourceWidth * sourceHeight * 4) {
      throw ArgumentError(
        'Taille des bytes RGBA invalide. Reçu: ${rgbaBytes.length}, attendu: ${sourceWidth * sourceHeight * 4}',
      );
    }

    int cursor = 0;

    for (int y = 0; y < targetHeight; y++) {
      final int sourceY =
          (y * sourceHeight / targetHeight).floor().clamp(0, sourceHeight - 1);

      for (int x = 0; x < targetWidth; x++) {
        final int sourceX =
            (x * sourceWidth / targetWidth).floor().clamp(0, sourceWidth - 1);

        final int pixelIndex = (sourceY * sourceWidth + sourceX) * 4;

        final int red = rgbaBytes[pixelIndex];
        final int green = rgbaBytes[pixelIndex + 1];
        final int blue = rgbaBytes[pixelIndex + 2];

        if (targetChannels == 1) {
          final double luminance =
              ((0.299 * red) + (0.587 * green) + (0.114 * blue)) / 255.0;
          outputBuffer[cursor++] = _normalize(luminance, normalizationMode);
        } else {
          outputBuffer[cursor++] = _normalize(red / 255.0, normalizationMode);
          outputBuffer[cursor++] = _normalize(green / 255.0, normalizationMode);
          outputBuffer[cursor++] = _normalize(blue / 255.0, normalizationMode);

          for (int channel = 3; channel < targetChannels; channel++) {
            outputBuffer[cursor++] = 0;
          }
        }
      }
    }
  }

  static void _validateTargetDimensions({
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
  }) {
    if (targetWidth <= 0 || targetHeight <= 0 || targetChannels <= 0) {
      throw ArgumentError('Dimensions cible invalides pour le prétraitement.');
    }
  }

  static double _normalize(
    double zeroToOneValue,
    InputNormalizationMode mode,
  ) {
    if (mode == InputNormalizationMode.mobileNetV2) {
      return (zeroToOneValue * 2.0) - 1.0;
    }
    return zeroToOneValue;
  }
}
