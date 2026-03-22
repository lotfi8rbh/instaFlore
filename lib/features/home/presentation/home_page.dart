import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../flower_detection/presentation/flower_detection_page.dart';
import '../../onboarding/presentation/onboarding_page.dart';
import '../../photo_detection/presentation/photo_detection_page.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('InstaFlore'),
        actions: <Widget>[
          IconButton(
            icon: const Icon(Icons.help_outline),
            tooltip: 'Comment ça marche ?',
            onPressed: () => _showHowItWorks(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            // Hero
            Container(
              padding: const EdgeInsets.fromLTRB(24, 40, 24, 40),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: <Color>[
                    colors.primaryContainer,
                    colors.surface,
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
              child: Column(
                children: <Widget>[
                  Icon(
                    Icons.local_florist,
                    size: 72,
                    color: colors.primary,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Identifiez vos fleurs',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: colors.onPrimaryContainer,
                        ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'En temps réel ou depuis votre galerie,\nobtenez le nom de votre fleur instantanément.',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                          color: colors.onPrimaryContainer.withOpacity(0.8),
                        ),
                  ),
                ],
              ),
            ),

            // Actions
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 28, 20, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: <Widget>[
                  _ActionCard(
                    icon: Icons.camera_alt_outlined,
                    title: 'Caméra en direct',
                    subtitle: 'Pointez votre caméra vers une fleur',
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => const FlowerDetectionPage(),
                      ),
                    ),
                    filled: true,
                  ),
                  const SizedBox(height: 12),
                  _ActionCard(
                    icon: Icons.photo_library_outlined,
                    title: 'Analyser une photo',
                    subtitle: 'Choisissez une image depuis la galerie',
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => const PhotoDetectionPage(),
                      ),
                    ),
                    filled: false,
                  ),
                  const SizedBox(height: 32),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showHowItWorks(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (BuildContext ctx) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Theme.of(ctx).colorScheme.outlineVariant,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'Comment ça marche ?',
                style: Theme.of(ctx).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 16),
              const _HowItWorksStep(
                number: '1',
                text: 'Choisissez le mode caméra ou galerie.',
              ),
              const SizedBox(height: 10),
              const _HowItWorksStep(
                number: '2',
                text: 'Cadrez bien la fleur, en lumière naturelle de préférence.',
              ),
              const SizedBox(height: 10),
              const _HowItWorksStep(
                number: '3',
                text: 'Le nom de la fleur s\'affiche automatiquement.',
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: TextButton(
                  onPressed: () async {
                    Navigator.of(ctx).pop();
                    final SharedPreferences prefs =
                        await SharedPreferences.getInstance();
                    await prefs.setBool(
                        OnboardingPage.completedKey, false);
                    if (!context.mounted) return;
                    Navigator.of(context).pushReplacement(
                      MaterialPageRoute<void>(
                        builder: (_) => const OnboardingPage(),
                      ),
                    );
                  },
                  child: const Text('Revoir l\'introduction'),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _ActionCard extends StatelessWidget {
  const _ActionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
    required this.filled,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;
  final bool filled;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;

    final Color bg =
        filled ? colors.primary : colors.surfaceContainerHighest;
    final Color fg = filled ? colors.onPrimary : colors.onSurface;

    return Material(
      color: bg,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
          child: Row(
            children: <Widget>[
              Icon(icon, size: 32, color: fg),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Text(
                      title,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: fg,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: TextStyle(
                        fontSize: 13,
                        color: fg.withOpacity(0.75),
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: fg.withOpacity(0.6)),
            ],
          ),
        ),
      ),
    );
  }
}

class _HowItWorksStep extends StatelessWidget {
  const _HowItWorksStep({required this.number, required this.text});

  final String number;
  final String text;

  @override
  Widget build(BuildContext context) {
    final ColorScheme colors = Theme.of(context).colorScheme;

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Container(
          width: 28,
          height: 28,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: colors.primaryContainer,
            shape: BoxShape.circle,
          ),
          child: Text(
            number,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: colors.onPrimaryContainer,
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(text, style: Theme.of(context).textTheme.bodyMedium),
          ),
        ),
      ],
    );
  }
}
