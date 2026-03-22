import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../flower_detection/presentation/flower_detection_page.dart';
import '../../onboarding/presentation/onboarding_page.dart';
import '../../photo_detection/presentation/photo_detection_page.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  static const List<String> _supportedClasses = <String>[
    'daisy',
    'dandelion',
    'roses',
    'sunflowers',
    'tulips',
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('InstaFlore'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Reconnaissance de fleurs en temps réel',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              'Pointe la caméra vers une fleur pour obtenir une prédiction instantanée.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: () {
                  Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const FlowerDetectionPage(),
                    ),
                  );
                },
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('Lancer la détection'),
              ),
            ),
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () {
                  Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const PhotoDetectionPage(),
                    ),
                  );
                },
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Analyser une photo'),
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: <Widget>[
                OutlinedButton.icon(
                  onPressed: () => _showQuickGuide(context),
                  icon: const Icon(Icons.menu_book_outlined),
                  label: const Text('Guide rapide'),
                ),
                OutlinedButton.icon(
                  onPressed: () => _showModelInfo(context),
                  icon: const Icon(Icons.analytics_outlined),
                  label: const Text('Infos modèle'),
                ),
                OutlinedButton.icon(
                  onPressed: () => _restartOnboarding(context),
                  icon: const Icon(Icons.refresh_outlined),
                  label: const Text('Revoir onboarding'),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Text(
                      'Classes actuellement supportées',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _supportedClasses
                          .map(
                            (String label) => Chip(label: Text(label)),
                          )
                          .toList(growable: false),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showQuickGuide(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      builder: (BuildContext context) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const <Widget>[
              Text(
                'Guide rapide',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              ),
              SizedBox(height: 10),
              Text('1. Place la fleur au centre de la caméra.'),
              SizedBox(height: 6),
              Text('2. Évite le contre-jour, privilégie une bonne lumière.'),
              SizedBox(height: 6),
              Text('3. Vérifie la confiance; si faible, rapproche la caméra.'),
            ],
          ),
        );
      },
    );
  }

  void _showModelInfo(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Informations modèle'),
          content: const Text(
            'Le modèle est optimisé pour la reconnaissance temps réel de 5 classes de fleurs.\n\n'
            'Conseil: privilégie une bonne lumière et une fleur bien centrée pour améliorer la fiabilité.',
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Fermer'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _restartOnboarding(BuildContext context) async {
    final SharedPreferences preferences = await SharedPreferences.getInstance();
    await preferences.setBool(OnboardingPage.completedKey, false);

    if (!context.mounted) {
      return;
    }

    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => const OnboardingPage(),
      ),
    );
  }
}
