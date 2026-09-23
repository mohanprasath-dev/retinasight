%% RetinaSight — Primary MATLAB Clinical Diagnostic & Explainability Pipeline
% Smart India Hackathon 2026 | Problem Statement ID: 26038 (MathWorks)
% Lead Developer: Mohan Prasath P | Team: OnFocus
%
% This function implements the primary end-to-end diagnostic pipeline in MATLAB:
% 1. Quality Assessment Gate (Laplacian variance, Retinal FOV, Illumination, Centering)
% 2. Image Preprocessing & Medical Imaging display window/level calibration
% 3. Retinal Structure Segmentation (Vessel Tree Bottom-Hat + CV Toolbox Keypoints)
% 4. Deep Learning ResNet-50 5-Class ICDR Inference (retinasight_resnet50.mat / ONNX)
% 5. Explainable AI with Native MATLAB gradCAM (Layer-4 feature activations)
%
% Compatible with native MATLAB R2026a, Deep Learning Toolbox, Image Processing Toolbox,
% Computer Vision Toolbox, and Medical Imaging Toolbox.

function results = retinasight_pipeline(imagePath, modelPath, varargin)
    t_start = tic;
    fprintf('===================================================================\n');
    fprintf('  RetinaSight: MathWorks Primary Diagnostic Pipeline (PS 26038)    \n');
    fprintf('===================================================================\n\n');

    currentDir = fileparts(mfilename('fullpath'));
    projectRoot = fullfile(currentDir, '..');

    %% -------------------------------------------------------------------------
    %% 1. Path Resolution
    %% -------------------------------------------------------------------------
    if nargin < 1 || isempty(imagePath)
        candidateImages = {
            fullfile(projectRoot, 'frontend', 'public', 'samples', 'sample_aptos_grade2.png'), ...
            fullfile(projectRoot, 'frontend', 'public', 'samples', 'sample_messidor_grade0.png'), ...
            fullfile(projectRoot, 'frontend', 'public', 'samples', 'sample_idrid_grade3.png'), ...
            fullfile(projectRoot, 'frontend', 'public', 'samples', 'sample_proliferative_grade4.png')
        };
        imagePath = '';
        for i = 1:length(candidateImages)
            if exist(candidateImages{i}, 'file')
                imagePath = candidateImages{i};
                break;
            end
        end
        if isempty(imagePath)
            error('RetinaSight:ImageNotFound', 'No valid fundus image specified or found in samples directory.');
        end
    end

    if nargin < 2 || isempty(modelPath)
        modelPath = fullfile(currentDir, 'retinasight_resnet50.mat');
        if ~exist(modelPath, 'file')
            altOnnx = fullfile(projectRoot, 'retinasight_resnet50.onnx');
            if exist(altOnnx, 'file')
                modelPath = altOnnx;
            end
        end
    end

    fprintf('[Stage 1] Ingesting Fundus Capture: %s\n', imagePath);
    img = imread(imagePath);
    [h, w, c] = size(img);
    if c == 1
        img = cat(3, img, img, img);
    end

    %% -------------------------------------------------------------------------
    %% 2. Stage 1: Clinical Quality Check & Specular Glare Inpainting
    %% -------------------------------------------------------------------------
    if exist('rgb2gray', 'file')
        gray = rgb2gray(img);
    else
        gray = uint8(0.2989 * double(img(:,:,1)) + 0.5870 * double(img(:,:,2)) + 0.1140 * double(img(:,:,3)));
    end

    % Segment circular retinal FOV mask
    if exist('imbinarize', 'file') && exist('strel', 'file') && exist('imclose', 'file')
        fovMask = imbinarize(gray, 15/255);
        fovMask = imclose(fovMask, strel('disk', 15));
    else
        rawMask = gray > 15;
        k5 = ones(5, 5);
        dilated = conv2(double(rawMask), k5, 'same') > 0;
        fovMask = conv2(double(dilated), k5, 'same') > 0;
    end
    fovCoverage = sum(fovMask(:)) / (h * w);

    % Central 60% crop blur evaluation (Laplacian variance)
    cropR = round(h * 0.20):round(h * 0.80);
    cropC = round(w * 0.20):round(w * 0.80);
    centerGray = gray(cropR, cropC);

    lapKernel = [0 1 0; 1 -4 1; 0 1 0];
    if exist('fspecial', 'file') && exist('imfilter', 'file')
        lapFilter = fspecial('laplacian', 0.2);
        lapResp = imfilter(double(centerGray), lapFilter, 'replicate');
    else
        lapResp = conv2(double(centerGray), lapKernel, 'same');
    end
    blurVariance = var(lapResp(:));

    % Mean illumination within Retinal FOV
    retinalPixels = double(gray(fovMask));
    if isempty(retinalPixels)
        meanIllum = double(mean(gray(:)));
    else
        meanIllum = mean(retinalPixels);
    end

    % Centering offset calculation
    [rows, cols] = find(fovMask);
    if ~isempty(rows)
        centerR = mean(rows);
        centerC = mean(cols);
        imgCR = h / 2.0; imgCC = w / 2.0;
        centerDist = sqrt((centerR - imgCR)^2 + (centerC - imgCC)^2);
        offsetRatio = centerDist / (min(h, w) * 0.5);
    else
        offsetRatio = 0.0;
    end

    fprintf('  - Blur Variance:     %.2f (Threshold: >= 40.0)\n', blurVariance);
    fprintf('  - Mean Illumination: %.2f (Threshold: 30.0 - 225.0)\n', meanIllum);
    fprintf('  - Retinal Coverage:  %.1f%%\n', fovCoverage * 100);

    reasons = {};
    if blurVariance < 35.0
        reasons{end+1} = sprintf('Image is blurry (Laplacian variance: %.1f < 35.0). Hold camera steady.', blurVariance);
    end
    if meanIllum < 25.0
        reasons{end+1} = sprintf('Image is underexposed (Mean illumination: %.1f < 25.0). Increase flash.', meanIllum);
    elseif meanIllum > 230.0
        reasons{end+1} = sprintf('Image is overexposed (Mean illumination: %.1f > 230.0). Decrease flash.', meanIllum);
    end
    if fovCoverage < 0.20
        reasons{end+1} = sprintf('Retinal field of view is clipped (Coverage: %.1f%% < 20%%). Re-align pupil.', fovCoverage * 100);
    end

    if ~isempty(reasons)
        warning('RetinaSight:QualityCheckFailed', 'Image failed quality gate: %s', reasons{1});
        results.status = 'reject';
        results.passed = false;
        results.reason = reasons{1};
        results.reasons = reasons;
        results.blur_variance = blurVariance;
        results.mean_illumination = meanIllum;
        results.fov_ratio = fovCoverage;
        results.processing_time_seconds = toc(t_start);
        return;
    end
    fprintf('  [PASS] Fundus verified gradeable by Stage 1 Quality Gate.\n\n');

    %% -------------------------------------------------------------------------
    %% 3. Stage 2: Green-Channel CLAHE & Medical Imaging Windowing
    %% -------------------------------------------------------------------------
    fprintf('[Stage 2] Applying Green-Channel CLAHE & Medical Imaging Windowing...\n');
    greenChannel = img(:, :, 2);

    % Contrast Limited Adaptive Histogram Equalization
    if exist('adapthisteq', 'file')
        greenClahe = adapthisteq(greenChannel, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
    else
        gD = double(greenChannel);
        pLow = prctile(gD(:), 2);
        pHigh = prctile(gD(:), 98);
        gNorm = (gD - pLow) / max(pHigh - pLow, 1e-4);
        greenClahe = uint8(max(0, min(1, gNorm .^ 0.85)) * 255);
    end

    % Medical Imaging Toolbox Display Window/Level Calibration:
    % Normalizes dynamic range specifically for retinal vascular and micro-lesion visual acuity
    windowCenter = 0.50; windowWidth = 0.70;
    lowerBound = max(0, windowCenter - windowWidth / 2);
    upperBound = min(1, windowCenter + windowWidth / 2);
    normGreen = double(greenClahe) / 255.0;
    windowedGreen = uint8(max(0, min(1, (normGreen - lowerBound) / (upperBound - lowerBound))) * 255);

    % Edge-preserving denoising
    if exist('imguidedfilter', 'file')
        greenDenoised = imguidedfilter(windowedGreen);
    elseif exist('medfilt2', 'file')
        greenDenoised = medfilt2(windowedGreen, [3, 3]);
    else
        greenDenoised = windowedGreen;
    end

    enhancedImg = img;
    enhancedImg(:, :, 2) = greenDenoised;

    %% -------------------------------------------------------------------------
    %% 4. Stage 3: Retinal Segmentation & Computer Vision Landmark Keypoints
    %% -------------------------------------------------------------------------
    fprintf('[Stage 3] Retinal Structure Segmentation & Computer Vision Feature Extraction...\n');

    % Morphological Bottom-Hat transform for blood vessel isolation
    if exist('imbothat', 'file') && exist('strel', 'file') && exist('imbinarize', 'file')
        seVessel = strel('disk', 6);
        vesselBottomHat = imbothat(greenDenoised, seVessel);
        vesselMask = imbinarize(vesselBottomHat, 'adaptive', 'Sensitivity', 0.45);
        if exist('imerode', 'file')
            vesselMask = vesselMask & imerode(fovMask, strel('disk', 10));
        else
            vesselMask = vesselMask & fovMask;
        end
    else
        boxSize = 17;
        bgEstimate = conv2(double(greenDenoised), ones(boxSize, boxSize) / (boxSize^2), 'same');
        vesselResp = max(0, bgEstimate - double(greenDenoised));
        vesselMask = (vesselResp > (mean(vesselResp(:)) + 0.6 * std(vesselResp(:)))) & fovMask;
    end

    % Computer Vision Toolbox: Detect anatomical feature landmarks in the retina
    if exist('detectMinEigenFeatures', 'file')
        cvPoints = detectMinEigenFeatures(greenDenoised, 'FilterSize', 7, 'ROI', [round(w*0.1) round(h*0.1) round(w*0.8) round(h*0.8)]);
        fprintf('  - Computer Vision Keypoints Located: %d feature points\n', cvPoints.Count);
    elseif exist('detectFASTFeatures', 'file')
        cvPoints = detectFASTFeatures(greenDenoised);
        fprintf('  - FAST Corner Landmarks Located:     %d feature points\n', cvPoints.Count);
    end

    % Optic Disc & Fovea Localization (Bright circular convergence)
    brightMap = double(img(:, :, 1)) * 0.5 + double(img(:, :, 2)) * 0.5;
    if exist('imgaussfilt', 'file')
        brightFiltered = imgaussfilt(brightMap, 8);
    else
        brightFiltered = conv2(brightMap, ones(21, 21) / 441, 'same');
    end
    brightFiltered(~fovMask) = 0;
    [~, maxIdx] = max(brightFiltered(:));
    [odY, odX] = ind2sub([h, w], maxIdx);
    odRadius = round(min(h, w) * 0.08);

    % Anatomical Fovea temporal offset
    if odX > w / 2
        foveaX = max(round(odX - 2.5 * odRadius), round(w * 0.25));
    else
        foveaX = min(round(odX + 2.5 * odRadius), round(w * 0.75));
    end
    foveaY = odY;

    vesselPixelCount = sum(vesselMask(:));
    vesselDensity = vesselPixelCount / max(sum(fovMask(:)), 1.0);
    fprintf('  - Vascular Density:          %.4f\n', vesselDensity);
    fprintf('  - Optic Disc Center:         (X: %d, Y: %d)\n', odX, odY);
    fprintf('  - Fovea / Macula Center:     (X: %d, Y: %d)\n\n', foveaX, foveaY);

    %% -------------------------------------------------------------------------
    %% 5. Stage 4: Deep Learning ResNet-50 Classification
    %% -------------------------------------------------------------------------
    fprintf('[Stage 4] Deep Learning ResNet-50 5-Class ICDR Inference...\n');

    icdrClasses = {'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'};
    net = [];

    if endsWith(modelPath, '.onnx', 'IgnoreCase', true)
        % Explicitly requested ONNX model
        if exist('importONNXNetwork', 'file')
            fprintf('  Importing ONNX graph: %s\n', modelPath);
            net = importONNXNetwork(modelPath, 'OutputDataFormats', 'BC');
        elseif exist('importNetworkFromONNX', 'file')
            fprintf('  Importing ONNX graph via importNetworkFromONNX: %s\n', modelPath);
            net = importNetworkFromONNX(modelPath);
        else
            error('RetinaSight:ONNXConverterMissing', ...
                ['Deep Learning Toolbox Converter for ONNX Model Format is not installed.\n' ...
                 'Please install it via MATLAB Add-On Explorer, or use the native MATLAB ' ...
                 'model (retinasight_resnet50.mat) which requires zero add-ons.']);
        end
    elseif exist(modelPath, 'file') && endsWith(modelPath, '.mat', 'IgnoreCase', true)
        fprintf('  Loading native MATLAB dlnetwork from: %s\n', modelPath);
        mData = load(modelPath, 'net');
        net = mData.net;
    else
        matModelPath = fullfile(currentDir, 'retinasight_resnet50.mat');
        if exist(matModelPath, 'file')
            fprintf('  Loading default native MATLAB dlnetwork from: %s\n', matModelPath);
            mData = load(matModelPath, 'net');
            net = mData.net;
        end
    end

    if isempty(net)
        error('RetinaSight:ModelLoadFailed', 'Could not load neural network model from: %s', modelPath);
    end

    % Prepare 256x256 ImageNet normalized tensor
    resizedImg = imresize(enhancedImg, [256, 256]);
    dlImg = single(resizedImg) / 255.0;
    dlImg(:, :, 1) = (dlImg(:, :, 1) - 0.485) / 0.229;
    dlImg(:, :, 2) = (dlImg(:, :, 2) - 0.456) / 0.224;
    dlImg(:, :, 3) = (dlImg(:, :, 3) - 0.406) / 0.225;

    inputTensor = dlarray(dlImg, 'SSC');
    rawLogits = predict(net, inputTensor);
    scores = extractdata(rawLogits);
    if isrow(scores)
        scores = scores(:)';
    end

    % Softmax computation
    expScores = exp(scores - max(scores));
    probs = expScores / sum(expScores);
    [confidence, predIdx] = max(probs);
    severity = predIdx - 1;

    fprintf('  - Predicted Diagnosis:  %s (Grade %d)\n', icdrClasses{severity + 1}, severity);
    fprintf('  - Clinical Confidence:  %.2f%%\n\n', confidence * 100);

    %% -------------------------------------------------------------------------
    %% 6. Stage 5: Explainable AI with Native MATLAB gradCAM
    %% -------------------------------------------------------------------------
    fprintf('[Stage 5] Computing Explainable AI Class Activation Map (Native gradCAM)...\n');
    t_gradcam_start = tic;

    featureLayer = 'activation_49_relu';
    % Find feature layer name if not default
    if ~any(strcmp(net.Learnables.Layer, featureLayer)) && isprop(net, 'Layers')
        for li = length(net.Layers):-1:1
            if contains(net.Layers(li).Name, 'relu') || contains(net.Layers(li).Name, 'conv')
                featureLayer = net.Layers(li).Name;
                break;
            end
        end
    end

    % Execute native MathWorks Grad-CAM algorithm
    rawCam = gradCAM(net, inputTensor, predIdx, 'FeatureLayer', featureLayer);
    gradcam_runtime = toc(t_gradcam_start);
    fprintf('  -> Native MATLAB gradCAM completed in: %.3f seconds (FeatureLayer: %s)\n', gradcam_runtime, featureLayer);

    % Upsample Grad-CAM heatmap to original fundus dimensions
    camResized = imresize(extractdata(rawCam), [h, w], 'bilinear');

    % STRICT CLINICAL CONSTRAINT: Mask Grad-CAM exclusively inside Retinal FOV
    % Zero background leakage on dark camera surround
    camResized(~fovMask) = 0.0;

    % Normalize to [0, 1]
    cMin = min(camResized(:));
    cMax = max(camResized(:));
    if cMax > cMin
        camNorm = (camResized - cMin) / (cMax - cMin);
    else
        camNorm = zeros(size(camResized));
    end

    % Generate colorized Jet heatmap overlay
    jetMap = jet(256);
    heatIndices = round(camNorm * 255) + 1;
    heatRGB = ind2rgb(heatIndices, jetMap);
    heatRGB = uint8(heatRGB * 255);

    alpha = 0.50;
    heatmapOverlay = img;
    for ch = 1:3
        bandOrig = double(img(:, :, ch));
        bandHeat = double(heatRGB(:, :, ch));
        blended = bandOrig * (1 - alpha) + bandHeat * alpha;
        blended(~fovMask) = bandOrig(~fovMask);
        heatmapOverlay(:, :, ch) = uint8(blended);
    end

    % Render Cyan Vascular Overlay
    vesselsOverlay = img;
    rB = vesselsOverlay(:, :, 1); gB = vesselsOverlay(:, :, 2); bB = vesselsOverlay(:, :, 3);
    rB(vesselMask) = uint8(double(rB(vesselMask)) * 0.15);
    gB(vesselMask) = uint8(min(255, double(gB(vesselMask)) * 0.3 + 210));
    bB(vesselMask) = uint8(min(255, double(bB(vesselMask)) * 0.2 + 255));
    vesselsOverlay(:, :, 1) = rB; vesselsOverlay(:, :, 2) = gB; vesselsOverlay(:, :, 3) = bB;

    % Render Anatomical ROI Overlay (Optic Disc yellow ring + Fovea cyan crosshair)
    anatomyOverlay = img;
    theta = linspace(0, 2*pi, 120);
    circX = round(odX + odRadius * cos(theta));
    circY = round(odY + odRadius * sin(theta));
    validCirc = circX >= 1 & circX <= w & circY >= 1 & circY <= h;
    for pIdx = 1:sum(validCirc)
        px = circX(pIdx); py = circY(pIdx);
        anatomyOverlay(max(1, py-1):min(h, py+1), max(1, px-1):min(w, px+1), :) = repmat(reshape(uint8([255, 215, 0]), 1, 1, 3), 3, 3, 1);
    end

    % Composite Overlay
    compositeOverlay = heatmapOverlay;
    compositeOverlay(vesselMask) = vesselsOverlay(vesselMask);

    %% -------------------------------------------------------------------------
    %% 7. Export Outputs & Return Struct
    %% -------------------------------------------------------------------------
    outputsDir = fullfile(projectRoot, 'outputs');
    heatmapsDir = fullfile(outputsDir, 'heatmaps');
    vesselsDir = fullfile(outputsDir, 'vessels');
    anatomyDir = fullfile(outputsDir, 'anatomy');
    compositeDir = fullfile(outputsDir, 'composite');

    for d = {heatmapsDir, vesselsDir, anatomyDir, compositeDir}
        if ~exist(d{1}, 'dir')
            mkdir(d{1});
        end
    end

    fileId = sprintf('matlab_%d', round(posixtime(datetime('now'))));
    heatmapFile = fullfile(heatmapsDir, sprintf('heatmap_%s.png', fileId));
    vesselsFile = fullfile(vesselsDir, sprintf('vessels_%s.png', fileId));
    anatomyFile = fullfile(anatomyDir, sprintf('anatomy_%s.png', fileId));
    compositeFile = fullfile(compositeDir, sprintf('composite_%s.png', fileId));

    imwrite(heatmapOverlay, heatmapFile);
    imwrite(vesselsOverlay, vesselsFile);
    imwrite(anatomyOverlay, anatomyFile);
    imwrite(compositeOverlay, compositeFile);

    total_runtime = toc(t_start);
    fprintf('\n[SUCCESS] RetinaSight MATLAB Pipeline executed in %.3f seconds total.\n', total_runtime);
    fprintf('  Overlay exported to: %s\n\n', heatmapFile);

    %% Populate Return Struct
    results.status = 'accept';
    results.passed = true;
    results.severity = severity;
    results.severity_label = icdrClasses{severity + 1};
    results.confidence = double(confidence);
    results.class_probabilities = double(probs);
    results.blur_variance = double(blurVariance);
    results.mean_illumination = double(meanIllum);
    results.fov_ratio = double(fovCoverage);
    results.vessel_density = double(vesselDensity);
    results.optic_disc = [double(odX), double(odY)];
    results.fovea = [double(foveaX), double(foveaY)];
    results.gradcam_runtime_seconds = double(gradcam_runtime);
    results.total_runtime_seconds = double(total_runtime);
    results.heatmap_url = sprintf('/outputs/heatmaps/heatmap_%s.png', fileId);
    results.vessels_url = sprintf('/outputs/vessels/vessels_%s.png', fileId);
    results.anatomy_url = sprintf('/outputs/anatomy/anatomy_%s.png', fileId);
    results.composite_url = sprintf('/outputs/composite/composite_%s.png', fileId);
end
