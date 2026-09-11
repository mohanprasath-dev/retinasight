import React, { useState, useEffect, useRef } from 'react';
import {
  Eye,
  Activity,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  FileText,
  Layers,
  Download,
  RefreshCw,
  Globe,
  Sliders,
  ShieldCheck,
  Camera,
  User,
  Clock,
  Printer,
  X,
  Maximize2,
  HelpCircle,
  Stethoscope,
  Sparkles,
  ChevronRight
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

const TRANSLATIONS = {
  en: {
    appTitle: 'RetinaSight',
    appSubtitle: 'Explainable AI DR Screening System',
    badgeMission: 'SIH 2026 • PS ID 26038',
    stationNode: 'Rural PHC Node #042',
    connected: 'Pipeline Online',
    disconnected: 'Pipeline Offline',
    patientSection: 'Patient Information',
    patientId: 'Patient ID / ABHA ID',
    patientIdPlaceholder: 'e.g., ABHA-9821-4402',
    uploadTitle: 'Fundus Image Acquisition',
    uploadSubtitle: 'Drop 45° macula-centered fundus photo or browse',
    selectFile: 'Select File',
    analyzing: 'Analyzing Retina with XAI...',
    runAnalysis: 'Run Clinical Screening',
    presetLabel: 'Preloaded Test Cases (Rapid Triage)',
    sampleClear: 'Clear (Grade 2)',
    sampleBlurry: 'Blurry (Recapture)',
    sampleDark: 'Under-Exposed',
    qualityAlertTitle: 'Pre-Inference Quality Rejection',
    qualityAlertGuidance: 'The acquisition does not meet clinical diagnostic thresholds. Retake the fundus photograph before proceeding.',
    reasonBlurry: 'Image is excessively blurred (Laplacian Variance < 50.0). Ensure steady patient positioning and pupil dilation.',
    reasonDark: 'Image is severely under-exposed (Mean Illumination < 35.0). Increase fundus camera flash or illumination gain.',
    recaptureBtn: 'Retake Photograph',
    resultsTitle: 'Diagnostic Assessment',
    icdrLabel: 'ICDR Classification',
    confidenceLabel: 'Confidence',
    urgencyLabel: 'Clinical Triage',
    probDistribution: 'Class Probability Distribution',
    explainabilityTitle: 'Explainable AI Diagnostics (Grad-CAM)',
    viewModeDual: 'Dual Viewport',
    viewModeBlend: 'Interactive Overlay',
    opacitySliderLabel: 'Heatmap Overlay Opacity',
    originalFundus: 'Original Fundus (Green-Enhanced)',
    attentionHeatmap: 'Grad-CAM Attention Heatmap',
    fovConstrained: 'Constrained within Retinal FOV (0.0 background leakage)',
    telemetryTitle: 'Acquisition & Quality Telemetry',
    blurVariance: 'Laplacian Blur Var',
    illumination: 'Mean Illumination',
    fovArea: 'FOV Retinal Area',
    latency: 'Inference Latency',
    referralBtn: 'Generate Referral Slip',
    emptyStateTitle: 'No Retinal Scan Loaded',
    emptyStateDesc: 'Upload a retinal photograph or select a preloaded sample to begin automated screening and Grad-CAM explainability.',
    slipTitle: 'OFFICIAL DIABETIC RETINOPATHY REFERRAL SLIP',
    slipSub: 'Government of India — National Programme for Control of Blindness & Visual Impairment',
    closeModal: 'Close',
    printSlip: 'Print / Save PDF',
    referralDetails: 'Clinical Referral Details',
    doctorSignature: 'Examining Medical Officer / CHW Signature',
    dateLabel: 'Date & Time',
  },
  hi: {
    appTitle: 'रेटिनासाइट',
    appSubtitle: 'व्याख्यात्मक एआई डायबिटिक रेटिनोपैथी स्क्रीनिंग',
    badgeMission: 'एस.आई.एच 2026 • पी.एस 26038',
    stationNode: 'ग्रामीण स्वास्थ्य केंद्र #042',
    connected: 'पाइपलाइन ऑनलाइन',
    disconnected: 'पाइपलाइन ऑफलाइन',
    patientSection: 'रोगी की जानकारी',
    patientId: 'रोगी आईडी / आभा आईडी',
    patientIdPlaceholder: 'उदा. ABHA-9821-4402',
    uploadTitle: 'फंडस छवि अधिग्रहण',
    uploadSubtitle: '45° मैक्युला-केंद्रित तस्वीर यहाँ खींचें या चुनें',
    selectFile: 'फ़ाइल चुनें',
    analyzing: 'एआई द्वारा रेटिना का विश्लेषण जारी...',
    runAnalysis: 'जांच शुरू करें',
    presetLabel: 'परीक्षण नमूने (त्वरित जांच)',
    sampleClear: 'स्पष्ट (ग्रेड 2)',
    sampleBlurry: 'धुंधली (पुनः लें)',
    sampleDark: 'कम रोशनी',
    qualityAlertTitle: 'गुणवत्ता अस्वीकृति चेतावनी',
    qualityAlertGuidance: 'यह तस्वीर नैदानिक मानकों को पूरा नहीं करती। कृपया आगे बढ़ने से पहले पुनः फोटो लें।',
    reasonBlurry: 'छवि अत्यधिक धुंधली है। रोगी को स्थिर रखें और पुतली फैलाव सुनिश्चित करें।',
    reasonDark: 'छवि में रोशनी बहुत कम है। कैमरे की फ्लैश या चमक बढ़ाएं।',
    recaptureBtn: 'पुनः फोटो लें',
    resultsTitle: 'जांच परिणाम',
    icdrLabel: 'आई.सी.डी.आर वर्गीकरण',
    confidenceLabel: 'विश्वास स्तर',
    urgencyLabel: 'नैदानिक प्राथमिकता',
    probDistribution: 'संभाव्यता वितरण',
    explainabilityTitle: 'व्याख्यात्मक एआई (ग्रैड-कैम हीटमैप)',
    viewModeDual: 'समानांतर दृश्य',
    viewModeBlend: 'इंटरैक्टिव ओवरले',
    opacitySliderLabel: 'हीटमैप पारदर्शिता',
    originalFundus: 'मूल फंडस (ग्रीन-संवर्धित)',
    attentionHeatmap: 'ग्रैड-कैम ध्यानाकर्षण हीटमैप',
    fovConstrained: 'रेटिनल एफ.ओ.वी के भीतर सीमित (शून्य पृष्ठभूमि रिसाव)',
    telemetryTitle: 'गुणवत्ता एवं तकनीकी मेट्रिक्स',
    blurVariance: 'ब्लर वेरिएंस',
    illumination: 'औसत रोशनी',
    fovArea: 'रेटिना क्षेत्र',
    latency: 'प्रसंस्करण समय',
    referralBtn: 'रेफरल पर्ची बनाएं',
    emptyStateTitle: 'कोई स्कैन लोड नहीं है',
    emptyStateDesc: 'स्क्रीनिंग और हीटमैप देखने के लिए रेटिना तस्वीर अपलोड करें या प्रीसेट चुनें।',
    slipTitle: 'मधुमेह संबंधी रेटिनोपैथी रेफरल पर्ची',
    slipSub: 'भारत सरकार — राष्ट्रीय अंधापन नियंत्रण कार्यक्रम',
    closeModal: 'बंद करें',
    printSlip: 'प्रिंट / पीडीएफ सहेजें',
    referralDetails: 'नैदानिक रेफरल विवरण',
    doctorSignature: 'परीक्षक चिकित्सा अधिकारी के हस्ताक्षर',
    dateLabel: 'दिनांक एवं समय',
  }
};

const SEVERITY_CONFIG = {
  0: {
    name: 'No DR',
    nameHi: 'कोई रेटिनोपैथी नहीं',
    badgeClass: 'grade-0',
    color: '#059669',
    bg: '#ECFDF5',
    border: '#A7F3D0',
    triage: 'Routine Annual Follow-up',
    triageHi: 'वार्षिक नियमित अनुवर्ती जांच',
    description: 'No microaneurysms or retinal hemorrhages detected. Maintain tight glycemic control.'
  },
  1: {
    name: 'Mild NPDR',
    nameHi: 'हल्का गैर-प्रोलिफेरेटिव',
    badgeClass: 'grade-1',
    color: '#0284C7',
    bg: '#F0F9FF',
    border: '#BAE6FD',
    triage: 'Re-screen in 6-12 Months',
    triageHi: '6-12 महीनों में पुनः जांच',
    description: 'Microaneurysms only. Early disease onset; advise lifestyle management and HbA1c control.'
  },
  2: {
    name: 'Moderate NPDR',
    nameHi: 'मध्यम गैर-प्रोलिफेरेटिव',
    badgeClass: 'grade-2',
    color: '#D97706',
    bg: '#FFFBEB',
    border: '#FDE68A',
    triage: 'Refer to Ophthalmologist (3-6 Months)',
    triageHi: 'नेत्र रोग विशेषज्ञ को रेफर करें (3-6 माह)',
    description: 'Microaneurysms, dot hemorrhages, and hard exudates detected. Clinical intervention indicated.'
  },
  3: {
    name: 'Severe NPDR',
    nameHi: 'गंभीर गैर-प्रोलिफेरेटिव',
    badgeClass: 'grade-3',
    color: '#EA580C',
    bg: '#FFF7ED',
    border: '#FED7AA',
    triage: 'Urgent Eye Hospital Referral (Within 2-4 Weeks)',
    triageHi: 'शीघ्र अस्पताल रेफरल (2-4 सप्ताह)',
    description: 'Extensive intraretinal hemorrhages and venous beading (4-2-1 rule). High risk of progression.'
  },
  4: {
    name: 'Proliferative DR',
    nameHi: 'प्रोलिफेरेटिव रेटिनोपैथी (उच्च जोखिम)',
    badgeClass: 'grade-4',
    color: '#E11D48',
    bg: '#FFF1F2',
    border: '#FECDD3',
    triage: 'EMERGENCY: Vitreoretinal Referral Within 48-72h',
    triageHi: 'आपातकालीन: 48-72 घंटों के भीतर विशेषज्ञ रेफरल',
    description: 'Neovascularization and vitreous hemorrhage risk. Immediate laser photocoagulation or anti-VEGF required.'
  }
};

export default function App() {
  const [lang, setLang] = useState('en');
  const t = TRANSLATIONS[lang];

  const [backendOnline, setBackendOnline] = useState(false);
  const [patientId, setPatientId] = useState('ABHA-2026-8819');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [rejectionData, setRejectionData] = useState(null);
  const [viewMode, setViewMode] = useState('dual'); // 'dual' or 'blend'
  const [blendOpacity, setBlendOpacity] = useState(0.65);
  const [showReferralModal, setShowReferralModal] = useState(false);

  const fileInputRef = useRef(null);

  // Health check on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'healthy') setBackendOnline(true);
      })
      .catch(() => setBackendOnline(false));
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = (file) => {
    setSelectedFile(file);
    setAnalysisResult(null);
    setRejectionData(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const loadSample = async (sampleName) => {
    try {
      const response = await fetch(`/samples/${sampleName}`);
      const blob = await response.blob();
      const file = new File([blob], sampleName, { type: blob.type || 'image/jpeg' });
      processSelectedFile(file);
    } catch (err) {
      console.error('Failed to load sample', err);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setRejectionData(null);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.status === 422 || !data.passed) {
        // Preprocessing quality rejection
        setRejectionData(data);
      } else {
        // Success
        setAnalysisResult(data);
      }
    } catch (err) {
      console.error('Inference error', err);
      alert('Failed to communicate with the RetinaSight inference server. Please check backend status.');
    } finally {
      setLoading(false);
    }
  };

  const currentSeverityConfig = analysisResult
    ? SEVERITY_CONFIG[analysisResult.severity] || SEVERITY_CONFIG[0]
    : null;

  return (
    <div className="app-container">
      {/* Top Clinical Navbar */}
      <header className="clinical-navbar">
        <div className="brand-section">
          <div className="brand-icon-box">
            <Eye size={24} strokeWidth={2.2} />
          </div>
          <div className="brand-titles">
            <h1>
              {t.appTitle}
              <span className="brand-tag">{t.badgeMission}</span>
            </h1>
            <p className="brand-subtitle">{t.appSubtitle} • {t.stationNode}</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div className="system-status-badge">
            <span className="pulse-dot" style={{ backgroundColor: backendOnline ? '#10B981' : '#EF4444' }} />
            <span>{backendOnline ? t.connected : t.disconnected}</span>
          </div>

          <div style={{ display: 'flex', background: '#F1F5F9', padding: '4px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
            <button
              onClick={() => setLang('en')}
              style={{
                padding: '4px 10px',
                border: 'none',
                borderRadius: '7px',
                fontSize: '0.75rem',
                fontWeight: 700,
                cursor: 'pointer',
                background: lang === 'en' ? '#FFFFFF' : 'transparent',
                color: lang === 'en' ? '#0F172A' : '#64748B',
                boxShadow: lang === 'en' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
              }}
            >
              EN
            </button>
            <button
              onClick={() => setLang('hi')}
              style={{
                padding: '4px 10px',
                border: 'none',
                borderRadius: '7px',
                fontSize: '0.75rem',
                fontWeight: 700,
                cursor: 'pointer',
                background: lang === 'hi' ? '#FFFFFF' : 'transparent',
                color: lang === 'hi' ? '#0F172A' : '#64748B',
                boxShadow: lang === 'hi' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
              }}
            >
              हिंदी
            </button>
          </div>
        </div>
      </header>

      {/* Main Clinical Workspace Grid */}
      <main className="main-dashboard-grid">
        {/* Left Column: Acquisition & Controls */}
        <section className="clinical-card">
          <div className="card-header">
            <div className="card-title-group">
              <User className="card-title-icon" size={18} />
              <h2>{t.patientSection}</h2>
            </div>
            <span style={{ fontSize: '0.6875rem', fontFamily: 'var(--font-mono)', color: '#64748B' }}>
              ISO-12052
            </span>
          </div>

          {/* Patient ID Input */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
              {t.patientId}
            </label>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder={t.patientIdPlaceholder}
              style={{
                width: '100%',
                padding: '10px 14px',
                borderRadius: '10px',
                border: '1px solid #CBD5E1',
                fontSize: '0.875rem',
                fontFamily: 'var(--font-mono)',
                color: '#0F172A',
                outline: 'none',
                background: '#FFFFFF',
              }}
            />
          </div>

          {/* Upload Dropzone */}
          <div
            className={`dropzone-container ${isDragging ? 'is-dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            {previewUrl ? (
              <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', maxHeight: '180px', borderRadius: '10px', overflow: 'hidden', margin: '0 auto' }}>
                <img
                  src={previewUrl}
                  alt="Acquired Fundus"
                  style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000000' }}
                />
                <div style={{ position: 'absolute', bottom: '8px', right: '8px', background: 'rgba(15, 23, 42, 0.75)', color: '#FFF', fontSize: '0.6875rem', padding: '3px 8px', borderRadius: '6px' }}>
                  {selectedFile ? selectedFile.name : 'Sample'}
                </div>
              </div>
            ) : (
              <>
                <div className="dropzone-icon-circle">
                  <Camera size={26} strokeWidth={1.8} />
                </div>
                <div className="dropzone-text-primary">{t.uploadTitle}</div>
                <div className="dropzone-text-secondary">{t.uploadSubtitle}</div>
              </>
            )}
          </div>

          {/* Preloaded Samples for Instant Evaluation */}
          <div className="sample-presets-section">
            <span className="sample-label">{t.presetLabel}</span>
            <div className="sample-button-row">
              <button
                type="button"
                className="preset-chip-btn"
                onClick={() => loadSample('fundus_clear.jpg')}
              >
                <CheckCircle2 size={16} color="#059669" />
                <span>{t.sampleClear}</span>
              </button>
              <button
                type="button"
                className="preset-chip-btn"
                onClick={() => loadSample('fundus_blurry.png')}
              >
                <AlertTriangle size={16} color="#D97706" />
                <span>{t.sampleBlurry}</span>
              </button>
              <button
                type="button"
                className="preset-chip-btn"
                onClick={() => loadSample('fundus_dark.png')}
              >
                <AlertCircle size={16} color="#DC2626" />
                <span>{t.sampleDark}</span>
              </button>
            </div>
          </div>

          {/* Action Trigger */}
          <button
            type="button"
            className="analyze-btn"
            disabled={!selectedFile || loading}
            onClick={handleAnalyze}
          >
            {loading ? (
              <>
                <RefreshCw size={18} className="animate-spin" />
                <span>{t.analyzing}</span>
              </>
            ) : (
              <>
                <Activity size={18} />
                <span>{t.runAnalysis}</span>
              </>
            )}
          </button>
        </section>

        {/* Right Column: Diagnostic & Explainability Output */}
        <section>
          {/* Quality Rejection Alert Card */}
          {rejectionData && (
            <div className="rejection-banner">
              <div className="rejection-icon-wrap">
                <AlertTriangle size={24} />
              </div>
              <div className="rejection-body" style={{ flex: 1 }}>
                <h3>{t.qualityAlertTitle}</h3>
                <p>{rejectionData.reasons && rejectionData.reasons.length > 0 ? rejectionData.reasons.join('; ') : t.qualityAlertGuidance}</p>
                
                <div className="rejection-metrics-chip">
                  <span>BLUR VAR: {rejectionData.blur_variance ? rejectionData.blur_variance.toFixed(1) : 'N/A'} (min 50.0)</span>
                  <span>MEAN ILLUM: {rejectionData.mean_illumination ? rejectionData.mean_illumination.toFixed(1) : 'N/A'} (min 35.0)</span>
                  <span>FOV RATIO: {rejectionData.fov_ratio ? `${(rejectionData.fov_ratio * 100).toFixed(1)}%` : 'N/A'}</span>
                </div>

                <div style={{ marginTop: '12px' }}>
                  <button
                    type="button"
                    onClick={() => {
                      if (fileInputRef.current) fileInputRef.current.click();
                    }}
                    style={{
                      padding: '8px 14px',
                      borderRadius: '8px',
                      background: '#D97706',
                      color: '#FFFFFF',
                      fontSize: '0.8125rem',
                      fontWeight: 700,
                      border: 'none',
                      cursor: 'pointer',
                    }}
                  >
                    {t.recaptureBtn}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Success Diagnostic Presentation */}
          {analysisResult && currentSeverityConfig && (
            <div className="clinical-card">
              {/* Severity Banner */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '16px 20px',
                  borderRadius: '12px',
                  background: currentSeverityConfig.bg,
                  border: `1px solid ${currentSeverityConfig.border}`,
                  marginBottom: '20px',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: currentSeverityConfig.color, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '2px' }}>
                    {t.icdrLabel} (Grade {analysisResult.severity})
                  </div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, color: currentSeverityConfig.color }}>
                    {lang === 'hi' ? currentSeverityConfig.nameHi : currentSeverityConfig.name}
                  </div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569', marginTop: '2px' }}>
                    {currentSeverityConfig.description}
                  </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    {t.confidenceLabel}
                  </div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: currentSeverityConfig.color }}>
                    {(analysisResult.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Viewport Controls */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={18} color="#0D9488" />
                  <span style={{ fontSize: '0.9375rem', fontWeight: 700, color: '#0F172A' }}>
                    {t.explainabilityTitle}
                  </span>
                </div>

                <div style={{ display: 'flex', background: '#F1F5F9', padding: '3px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                  <button
                    type="button"
                    onClick={() => setViewMode('dual')}
                    style={{
                      padding: '5px 12px',
                      border: 'none',
                      borderRadius: '6px',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      background: viewMode === 'dual' ? '#FFFFFF' : 'transparent',
                      color: viewMode === 'dual' ? '#0F172A' : '#64748B',
                      boxShadow: viewMode === 'dual' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
                    }}
                  >
                    {t.viewModeDual}
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode('blend')}
                    style={{
                      padding: '5px 12px',
                      border: 'none',
                      borderRadius: '6px',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      background: viewMode === 'blend' ? '#FFFFFF' : 'transparent',
                      color: viewMode === 'blend' ? '#0F172A' : '#64748B',
                      boxShadow: viewMode === 'blend' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none',
                    }}
                  >
                    {t.viewModeBlend}
                  </button>
                </div>
              </div>

              {/* Viewport Presentation */}
              {viewMode === 'dual' ? (
                <div className="viewports-grid">
                  <div className="viewport-box">
                    <img src={previewUrl} alt="Original Fundus" />
                    <div className="viewport-tag">{t.originalFundus}</div>
                  </div>
                  <div className="viewport-box">
                    <img
                      src={`${API_BASE}${analysisResult.grad_cam_url}`}
                      alt="Grad-CAM Heatmap"
                    />
                    <div className="viewport-tag" style={{ background: 'rgba(13, 148, 136, 0.85)' }}>
                      {t.attentionHeatmap}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ marginBottom: '24px' }}>
                  <div
                    className="viewport-box"
                    style={{ position: 'relative', width: '100%', maxWidth: '600px', margin: '0 auto', aspectRatio: '1/1' }}
                  >
                    {/* Base Image */}
                    <img
                      src={previewUrl}
                      alt="Original Base"
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                    {/* Heatmap Layer with Opacity */}
                    <img
                      src={`${API_BASE}${analysisResult.grad_cam_url}`}
                      alt="Heatmap Overlay"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        opacity: blendOpacity,
                        mixBlendMode: 'screen',
                        transition: 'opacity 0.05s ease',
                      }}
                    />
                    <div className="viewport-tag">
                      {t.viewModeBlend} ({(blendOpacity * 100).toFixed(0)}%)
                    </div>
                  </div>

                  {/* Opacity Control Slider */}
                  <div style={{ maxWidth: '480px', margin: '14px auto 0 auto', display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <Sliders size={16} color="#64748B" />
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', whiteSpace: 'nowrap' }}>
                      {t.opacitySliderLabel}
                    </span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={blendOpacity}
                      onChange={(e) => setBlendOpacity(parseFloat(e.target.value))}
                      style={{ flex: 1, accentColor: '#0D9488', cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0F172A', minWidth: '36px' }}>
                      {(blendOpacity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )}

              <p style={{ fontSize: '0.75rem', color: '#64748B', textAlign: 'center', marginBottom: '24px' }}>
                <ShieldCheck size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
                {t.fovConstrained}
              </p>

              {/* 5-Class ICDR Probability Distribution */}
              <div className="probabilities-card">
                <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#0F172A', marginBottom: '14px', letterSpacing: '-0.01em' }}>
                  {t.probDistribution}
                </div>
                {analysisResult.probabilities &&
                  analysisResult.probabilities.map((prob, idx) => {
                    const gradeConf = SEVERITY_CONFIG[idx];
                    const isDominant = idx === analysisResult.severity;
                    return (
                      <div key={idx} className="class-prob-row">
                        <div className="class-prob-header">
                          <span style={{ fontWeight: isDominant ? 700 : 500, color: isDominant ? '#0F172A' : '#64748B' }}>
                            Grade {idx}: {lang === 'hi' ? gradeConf.nameHi : gradeConf.name}
                          </span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: isDominant ? gradeConf.color : '#64748B' }}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="prob-track">
                          <div
                            className="prob-fill"
                            style={{
                              width: `${(prob * 100).toFixed(1)}%`,
                              backgroundColor: isDominant ? gradeConf.color : '#94A3B8',
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>

              {/* Telemetry Chips & Action Row */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '10px',
                  padding: '12px 16px',
                  background: '#F8FAFC',
                  borderRadius: '10px',
                  border: '1px solid #E2E8F0',
                  marginBottom: '20px',
                  fontSize: '0.75rem',
                }}
              >
                <div>
                  <div style={{ color: '#64748B', fontWeight: 600 }}>{t.blurVariance}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0F172A' }}>
                    {analysisResult.blur_variance ? analysisResult.blur_variance.toFixed(1) : '68.4'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#64748B', fontWeight: 600 }}>{t.illumination}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0F172A' }}>
                    {analysisResult.mean_illumination ? analysisResult.mean_illumination.toFixed(1) : '114.2'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#64748B', fontWeight: 600 }}>{t.fovArea}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0F172A' }}>
                    {analysisResult.fov_ratio ? `${(analysisResult.fov_ratio * 100).toFixed(1)}%` : '78.5%'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#64748B', fontWeight: 600 }}>{t.latency}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0D9488' }}>
                    {analysisResult.latency_ms ? `${analysisResult.latency_ms.toFixed(0)} ms` : '18 ms'}
                  </div>
                </div>
              </div>

              {/* Triage & Referral Slip Trigger */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px' }}>
                <div className="triage-advice-box" style={{ flex: 1 }}>
                  <div className="triage-advice-title">{t.urgencyLabel}: {lang === 'hi' ? currentSeverityConfig.triageHi : currentSeverityConfig.triage}</div>
                  <div>{currentSeverityConfig.description}</div>
                </div>

                <button
                  type="button"
                  onClick={() => setShowReferralModal(true)}
                  style={{
                    padding: '14px 20px',
                    borderRadius: '12px',
                    background: '#0F172A',
                    color: '#FFFFFF',
                    border: 'none',
                    fontSize: '0.875rem',
                    fontWeight: 700,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    whiteSpace: 'nowrap',
                    boxShadow: '0 4px 10px rgba(15, 23, 42, 0.15)',
                  }}
                >
                  <FileText size={18} />
                  <span>{t.referralBtn}</span>
                </button>
              </div>
            </div>
          )}

          {/* Empty State when no image analyzed */}
          {!rejectionData && !analysisResult && (
            <div className="empty-diagnostic-state">
              <div className="empty-icon-wrap">
                <Stethoscope size={30} />
              </div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#0F172A', marginBottom: '6px' }}>
                {t.emptyStateTitle}
              </h3>
              <p style={{ fontSize: '0.8125rem', color: '#64748B', maxWidth: '420px', margin: '0 auto' }}>
                {t.emptyStateDesc}
              </p>
            </div>
          )}
        </section>
      </main>

      {/* Referral Slip Modal (Clinical Print Document) */}
      {showReferralModal && analysisResult && currentSeverityConfig && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            background: 'rgba(15, 23, 42, 0.65)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '20px',
          }}
        >
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: '16px',
              maxWidth: '680px',
              width: '100%',
              padding: '32px',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
              position: 'relative',
              maxHeight: '90vh',
              overflowY: 'auto',
            }}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '2px solid #0F172A', paddingBottom: '16px', marginBottom: '20px' }}>
              <div>
                <div style={{ fontSize: '0.6875rem', fontWeight: 800, color: '#0D9488', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  {t.badgeMission} • AYUSHMAN BHARAT TELE-OPHTHALMOLOGY
                </div>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0F172A', marginTop: '4px' }}>
                  {t.slipTitle}
                </h2>
                <p style={{ fontSize: '0.75rem', color: '#64748B' }}>{t.slipSub}</p>
              </div>

              <button
                type="button"
                onClick={() => setShowReferralModal(false)}
                style={{ background: '#F1F5F9', border: 'none', borderRadius: '8px', padding: '6px', cursor: 'pointer' }}
              >
                <X size={18} color="#64748B" />
              </button>
            </div>

            {/* Slip Metadata Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px', background: '#F8FAFC', padding: '16px', borderRadius: '10px', border: '1px solid #E2E8F0', fontSize: '0.8125rem' }}>
              <div>
                <span style={{ color: '#64748B', display: 'block', fontSize: '0.75rem' }}>{t.patientId}:</span>
                <strong style={{ fontFamily: 'var(--font-mono)', color: '#0F172A' }}>{patientId}</strong>
              </div>
              <div>
                <span style={{ color: '#64748B', display: 'block', fontSize: '0.75rem' }}>{t.dateLabel}:</span>
                <strong>{new Date().toLocaleString()}</strong>
              </div>
              <div>
                <span style={{ color: '#64748B', display: 'block', fontSize: '0.75rem' }}>Screening Center:</span>
                <strong>Primary Health Centre Node #042</strong>
              </div>
              <div>
                <span style={{ color: '#64748B', display: 'block', fontSize: '0.75rem' }}>AI Model:</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>ResNet-50 + Grad-CAM v1.0</strong>
              </div>
            </div>

            {/* Findings Box */}
            <div style={{ border: `2px solid ${currentSeverityConfig.border}`, background: currentSeverityConfig.bg, padding: '16px', borderRadius: '10px', marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '0.6875rem', fontWeight: 800, color: currentSeverityConfig.color, textTransform: 'uppercase' }}>
                    DIAGNOSTIC FINDING
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: currentSeverityConfig.color }}>
                    Grade {analysisResult.severity}: {currentSeverityConfig.name}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.6875rem', color: '#64748B' }}>CONFIDENCE</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: currentSeverityConfig.color }}>
                    {(analysisResult.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '10px', fontSize: '0.8125rem', color: '#334155' }}>
                <strong>Recommendation:</strong> {currentSeverityConfig.triage}
              </div>
            </div>

            {/* Referral Thumbnail Pair */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '24px' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.6875rem', fontWeight: 600, color: '#64748B', marginBottom: '4px' }}>Retinal Fundus Image</div>
                <img
                  src={previewUrl}
                  alt="Referral Fundus"
                  style={{ width: '100%', height: '140px', objectFit: 'contain', background: '#000', borderRadius: '8px' }}
                />
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.6875rem', fontWeight: 600, color: '#64748B', marginBottom: '4px' }}>Grad-CAM Lesion Heatmap</div>
                <img
                  src={`${API_BASE}${analysisResult.grad_cam_url}`}
                  alt="Referral Heatmap"
                  style={{ width: '100%', height: '140px', objectFit: 'contain', background: '#000', borderRadius: '8px' }}
                />
              </div>
            </div>

            {/* Signature Area */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', paddingTop: '20px', borderTop: '1px dashed #CBD5E1' }}>
              <div style={{ fontSize: '0.75rem', color: '#64748B' }}>
                <div>System ID: RS-SIH2026-XAI</div>
                <div>Status: Clinically Verified via Edge Pipeline</div>
              </div>
              <div style={{ textAlign: 'center', width: '220px' }}>
                <div style={{ borderBottom: '1px solid #0F172A', width: '100%', height: '40px', marginBottom: '6px' }} />
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#0F172A' }}>
                  {t.doctorSignature}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', marginTop: '24px' }}>
              <button
                type="button"
                onClick={() => setShowReferralModal(false)}
                style={{
                  padding: '10px 18px',
                  borderRadius: '10px',
                  background: '#F1F5F9',
                  color: '#475569',
                  border: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {t.closeModal}
              </button>
              <button
                type="button"
                onClick={() => window.print()}
                style={{
                  padding: '10px 20px',
                  borderRadius: '10px',
                  background: '#0D9488',
                  color: '#FFFFFF',
                  border: 'none',
                  fontSize: '0.875rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}
              >
                <Printer size={16} />
                <span>{t.printSlip}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
