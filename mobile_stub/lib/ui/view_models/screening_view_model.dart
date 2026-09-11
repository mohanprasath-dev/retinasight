import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import '../../data/models/prediction_result.dart';
import '../../data/services/screening_api_service.dart';

/// ViewModel managing state for PHC mobile fundus acquisition.
class ScreeningViewModel extends ChangeNotifier {
  ScreeningViewModel({
    ScreeningApiService? apiService,
    ImagePicker? picker,
  })  : _apiService = apiService ?? ScreeningApiService(),
        _picker = picker ?? ImagePicker();

  final ScreeningApiService _apiService;
  final ImagePicker _picker;

  String _patientId = 'ABHA-2026-9021';
  String get patientId => _patientId;

  File? _capturedImage;
  File? get capturedImage => _capturedImage;

  bool _isLoading = false;
  bool get isLoading => _isLoading;

  PredictionResult? _result;
  PredictionResult? get result => _result;

  String? _errorMessage;
  String? get errorMessage => _errorMessage;

  void setPatientId(String id) {
    _patientId = id;
    notifyListeners();
  }

  Future<void> pickImage(ImageSource source) async {
    try {
      final picked = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 92,
      );

      if (picked != null) {
        _capturedImage = File(picked.path);
        _result = null;
        _errorMessage = null;
        notifyListeners();
      }
    } catch (e) {
      _errorMessage = 'Failed to acquire photo: $e';
      notifyListeners();
    }
  }

  Future<void> submitScreening() async {
    if (_capturedImage == null) return;

    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      final res = await _apiService.predictRetina(
        imageFile: _capturedImage!,
        patientId: _patientId,
      );
      _result = res;
    } catch (e) {
      _errorMessage = 'Submission failed: $e';
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void reset() {
    _capturedImage = null;
    _result = null;
    _errorMessage = null;
    notifyListeners();
  }
}
