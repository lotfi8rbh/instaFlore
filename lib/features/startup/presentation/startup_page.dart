import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../../core/widgets/app_loading_view.dart';
import '../../home/presentation/home_page.dart';
import '../../onboarding/presentation/onboarding_page.dart';

class StartupPage extends StatefulWidget {
  const StartupPage({super.key});

  @override
  State<StartupPage> createState() => _StartupPageState();
}

class _StartupPageState extends State<StartupPage> {
  bool _isLoading = true;
  bool _showOnboarding = true;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    final SharedPreferences preferences = await SharedPreferences.getInstance();
    final bool completed =
        preferences.getBool(OnboardingPage.completedKey) ?? false;

    if (!mounted) {
      return;
    }

    setState(() {
      _showOnboarding = !completed;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: AppLoadingView(
          message: 'Préparation de l\'application...',
        ),
      );
    }

    return _showOnboarding ? const OnboardingPage() : const HomePage();
  }
}
