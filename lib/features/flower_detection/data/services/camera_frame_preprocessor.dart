import 'dart:typed_data';

import 'package:camera/camera.dart';

class CameraFramePreprocessor {
  const CameraFramePreprocessor._();

  static Float32List preprocess(
    CameraImage image, {
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
  }) {
    if (targetWidth <= 0 || targetHeight <= 0 || targetChannels <= 0) {
      throw ArgumentError('Dimensions cible invalides pour le prétraitement.');
    }

    if (image.format.group == ImageFormatGroup.yuv420) {
      return _preprocessYuv420(
        image,
        targetWidth: targetWidth,
        targetHeight: targetHeight,
        targetChannels: targetChannels,
      );
    }

    if (image.format.group == ImageFormatGroup.bgra8888) {
      return _preprocessBgra8888(
        image,
        targetWidth: targetWidth,
        targetHeight: targetHeight,
        targetChannels: targetChannels,
      );
    }

    throw UnsupportedError(
      'Format caméra non supporté: ${image.format.group}',
    );
  }

  static Float32List _preprocessYuv420(
    CameraImage image, {
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
  }) {
    final Plane yPlane = image.planes.first;
    final Float32List output = Float32List(
      targetWidth * targetHeight * targetChannels,
    );

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
        final double normalized = luma / 255.0;

        for (int channel = 0; channel < targetChannels; channel++) {
          output[cursor++] = normalized;
        }
      }
    }

    return output;
  }

  static Float32List _preprocessBgra8888(
    CameraImage image, {
    required int targetWidth,
    required int targetHeight,
    required int targetChannels,
  }) {
    final Plane plane = image.planes.first;
    final int bytesPerPixel = plane.bytesPerPixel ?? 4;

    final Float32List output = Float32List(
      targetWidth * targetHeight * targetChannels,
    );

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
          output[cursor++] = luminance;
        } else {
          output[cursor++] = red / 255.0;
          output[cursor++] = green / 255.0;
          output[cursor++] = blue / 255.0;

          for (int channel = 3; channel < targetChannels; channel++) {
            output[cursor++] = 0;
          }
        }
      }
    }

    return output;
  }
}
