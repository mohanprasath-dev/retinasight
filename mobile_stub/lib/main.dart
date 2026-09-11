import 'package:flutter/material.dart';
import 'ui/view_models/screening_view_model.dart';
import 'ui/views/screening_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RetinaSightMobileApp());
}

class RetinaSightMobileApp extends StatefulWidget {
  const RetinaSightMobileApp({super.key});

  @override
  State<RetinaSightMobileApp> createState() => _RetinaSightMobileAppState();
}

class _RetinaSightMobileAppState extends State<RetinaSightMobileApp> {
  late final ScreeningViewModel _viewModel;

  @override
  void initState() {
    super.initState();
    _viewModel = ScreeningViewModel();
  }

  @override
  void dispose() {
    _viewModel.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RetinaSight Mobile',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0D9488),
          brightness: Brightness.light,
          primary: const Color(0xFF0D9488),
          surface: Colors.white,
        ),
        fontFamily: 'Roboto',
      ),
      home: ScreeningScreen(viewModel: _viewModel),
    );
  }
}
