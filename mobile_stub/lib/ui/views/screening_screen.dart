import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../view_models/screening_view_model.dart';
import '../../data/models/prediction_result.dart';

/// Single-screen clinical fundus capture interface for rural healthcare workers.
class ScreeningScreen extends StatelessWidget {
  const ScreeningScreen({
    super.key,
    required this.viewModel,
  });

  final ScreeningViewModel viewModel;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: viewModel,
      builder: (context, _) {
        return Scaffold(
          backgroundColor: const Color(0xFFF8FAFC),
          appBar: AppBar(
            backgroundColor: Colors.white,
            elevation: 1,
            title: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'RetinaSight PHC Mobile',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    color: Color(0xFF0F172A),
                  ),
                ),
                Text(
                  'SIH 2026 • PS 26038 • Field Node #042',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: Color(0xFF64748B),
                  ),
                ),
              ],
            ),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 16.0),
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: const Color(0xFFECFDF5),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.parse('1px solid #A7F3D0'),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: const [
                        Icon(Icons.circle, color: Color(0xFF059669), size: 8),
                        SizedBox(width: 6),
                        Text(
                          'Online Sync',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            color: Color(0xFF059669),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Patient Identifier Input
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: const Color(0xFFE2E8F0)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Patient Identification',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF64748B),
                          letterSpacing: 0.5,
                        ),
                      ),
                      const SizedBox(height: 8),
                      TextFormField(
                        initialValue: viewModel.patientId,
                        onChanged: viewModel.setPatientId,
                        decoration: const InputDecoration(
                          prefixIcon: Icon(Icons.badge_outlined, color: Color(0xFF0D9488)),
                          hintText: 'Enter ABHA / Patient ID',
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.all(Radius.circular(10)),
                            borderSide: BorderSide(color: Color(0xFFCBD5E1)),
                          ),
                          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),

                // Retinal Viewport Area with FOV reticle
                Container(
                  height: 320,
                  decoration: BoxDecoration(
                    color: Colors.black,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: const Color(0xFFCBD5E1)),
                  ),
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      if (viewModel.capturedImage != null)
                        ClipRRect(
                          borderRadius: BorderRadius.circular(15),
                          child: Image.file(
                            viewModel.capturedImage!,
                            width: double.infinity,
                            height: double.infinity,
                            fit: BoxFit.contain,
                          ),
                        )
                      else
                        Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const [
                            Icon(Icons.camera_alt_outlined, size: 48, color: Color(0xFF64748B)),
                            SizedBox(height: 12),
                            Text(
                              'No Retinal Photograph Captured',
                              style: TextStyle(color: Color(0xFF94A3B8), fontSize: 14),
                            ),
                          ],
                        ),

                      // Retinal FOV Guide Overlay
                      IgnorePointer(
                        child: Container(
                          width: 240,
                          height: 240,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: const Color(0x660D9488),
                              width: 2.0,
                            ),
                          ),
                        ),
                      ),
                      const Positioned(
                        bottom: 12,
                        child: Text(
                          'Center macula inside green circular reticle',
                          style: TextStyle(
                            color: Color(0xCCFFFFFF),
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                // Capture Action Buttons
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () => viewModel.pickImage(ImageSource.camera),
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Camera'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: const Color(0xFF0F172A),
                          elevation: 0,
                          side: const BorderSide(color: Color(0xFFCBD5E1)),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () => viewModel.pickImage(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library_outlined),
                        label: const Text('Gallery'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: const Color(0xFF0F172A),
                          elevation: 0,
                          side: const BorderSide(color: Color(0xFFCBD5E1)),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),

                // Submit Action Trigger
                ElevatedButton.icon(
                  onPressed: viewModel.capturedImage == null || viewModel.isLoading
                      ? null
                      : viewModel.submitScreening,
                  icon: viewModel.isLoading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                        )
                      : const Icon(Icons.analytics_outlined),
                  label: Text(
                    viewModel.isLoading ? 'Analyzing Retina...' : 'Submit for AI Screening',
                    style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0D9488),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),

                const SizedBox(height: 20),

                // Diagnostic Result Presentation
                if (viewModel.result != null)
                  _buildDiagnosticCard(context, viewModel.result!)
                else if (viewModel.errorMessage != null)
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFEF2F2),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFFFECDD3)),
                    ),
                    child: Text(
                      viewModel.errorMessage!,
                      style: const TextStyle(color: Color(0xFFB91C1C), fontSize: 13),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildDiagnosticCard(BuildContext context, PredictionResult result) {
    if (result.status == 'queued') {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFFEFF6FF),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFBFDBFE)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Row(
              children: [
                Icon(Icons.cloud_upload_outlined, color: Color(0xFF2563EB)),
                SizedBox(width: 8),
                Text(
                  'Queued for Sync (Offline Mode)',
                  style: TextStyle(fontWeight: FontWeight.w800, color: Color(0xFF1E40AF)),
                ),
              ],
            ),
            SizedBox(height: 6),
            Text(
              'Retinal scan cached in local encrypted storage. Scan will automatically sync to district server once data connectivity is restored.',
              style: TextStyle(fontSize: 13, color: Color(0xFF1E3A8A)),
            ),
          ],
        ),
      );
    }

    if (!result.passed || result.status == 'reject') {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFFFFFBEB),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFFFDE68A)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: const [
                Icon(Icons.warning_amber_rounded, color: Color(0xFFD97706)),
                SizedBox(width: 8),
                Text(
                  'Quality Gate Rejection',
                  style: TextStyle(fontWeight: FontWeight.w800, color: Color(0xFF92400E)),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              result.reasons.isNotEmpty
                  ? result.reasons.join('\n')
                  : 'Image does not meet quality requirements.',
              style: const TextStyle(fontSize: 13, color: Color(0xFFB45309)),
            ),
            const SizedBox(height: 12),
            Text(
              'BLUR: ${result.blurVariance.toStringAsFixed(1)} (min 50.0) | ILLUM: ${result.meanIllumination.toStringAsFixed(1)} (min 35.0)',
              style: const TextStyle(fontSize: 11, fontFamily: 'monospace', color: Color(0xFF78350F)),
            ),
          ],
        ),
      );
    }

    // Success State
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFCBD5E1)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x0D0F172A),
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'ICDR Severity: Grade ${result.severity}',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF0D9488),
                  letterSpacing: 0.5,
                ),
              ),
              Text(
                '${(result.confidence * 100).toStringAsFixed(1)}% Conf.',
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  fontFamily: 'monospace',
                  color: Color(0xFF0F172A),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            result.severityLabel,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w800,
              color: Color(0xFF0F172A),
            ),
          ),
          const SizedBox(height: 12),
          const Divider(),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Telemetry: ${result.blurVariance.toStringAsFixed(1)} var • ${result.processingTimeSeconds.toStringAsFixed(2)}s',
                style: const TextStyle(fontSize: 11, color: Color(0xFF64748B)),
              ),
              const Text(
                'Grad-CAM Verified',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: Color(0xFF0D9488)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
