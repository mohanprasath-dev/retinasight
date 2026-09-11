/// Diagnostic screening result model matching RetinaSight FastAPI schema.
class PredictionResult {
  final String status;
  final bool passed;
  final int severity;
  final String severityLabel;
  final double confidence;
  final String gradCamUrl;
  final Map<String, double> classProbabilities;
  final double blurVariance;
  final double meanIllumination;
  final double fovRatio;
  final List<String> reasons;
  final double processingTimeSeconds;

  const PredictionResult({
    required this.status,
    required this.passed,
    this.severity = 0,
    this.severityLabel = 'No DR',
    this.confidence = 0.0,
    this.gradCamUrl = '',
    this.classProbabilities = const {},
    this.blurVariance = 0.0,
    this.meanIllumination = 0.0,
    this.fovRatio = 0.0,
    this.reasons = const [],
    this.processingTimeSeconds = 0.0,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    final status = json['status'] as String? ?? 'reject';
    final passed = json['passed'] as bool? ?? (status == 'accept');
    final rawProbs = json['class_probabilities'] as Map<String, dynamic>? ?? {};

    final Map<String, double> probs = {};
    rawProbs.forEach((k, v) {
      probs[k] = (v as num).toDouble();
    });

    final qm = json['quality_metrics'] as Map<String, dynamic>? ?? {};

    return PredictionResult(
      status: status,
      passed: passed,
      severity: (json['severity'] as num?)?.toInt() ?? 0,
      severityLabel: json['severity_label'] as String? ?? 'No DR',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      gradCamUrl: json['grad_cam_url'] as String? ?? '',
      classProbabilities: probs,
      blurVariance: (json['blur_variance'] as num?)?.toDouble() ??
          (qm['blur_variance'] as num?)?.toDouble() ??
          0.0,
      meanIllumination: (json['mean_illumination'] as num?)?.toDouble() ??
          (qm['mean_brightness'] as num?)?.toDouble() ??
          0.0,
      fovRatio: (json['fov_ratio'] as num?)?.toDouble() ??
          (qm['fov_coverage'] as num?)?.toDouble() ??
          0.0,
      reasons: (json['reasons'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          (json['reason'] != null ? [json['reason'].toString()] : []),
      processingTimeSeconds:
          (json['processing_time_seconds'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
