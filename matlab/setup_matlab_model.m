%% RetinaSight — Build & Cache Native MATLAB ResNet-50 dlnetwork
% SIH 2026, PS ID 26038, Team OnFocus

function net = setup_matlab_model()
    fprintf('===================================================================\n');
    fprintf('  RetinaSight: Building Native MATLAB ResNet-50 dlnetwork Graph    \n');
    fprintf('===================================================================\n\n');

    currentDir = fileparts(mfilename('fullpath'));
    weightsFile = fullfile(currentDir, 'retinasight_resnet50_weights.mat');
    outputMat = fullfile(currentDir, 'retinasight_resnet50.mat');

    if ~exist(weightsFile, 'file')
        error('Weights file not found: %s. Run python scripts/export_matlab_model.py first.', weightsFile);
    end

    fprintf('[1/3] Loading calibrated classification weights: %s\n', weightsFile);
    wData = load(weightsFile);

    fprintf('[2/3] Constructing ResNet-50 LayerGraph (Native Deep Learning Toolbox)...\n');
    lgraph = resnet50('Weights', 'none');

    % Adapt input layer to 256x256 without default normalization
    newInput = imageInputLayer([256 256 3], 'Name', 'input_1', 'Normalization', 'none');
    lgraph = replaceLayer(lgraph, 'input_1', newInput);

    % Replace 1000-class head with calibrated 5-class ICDR head
    newFC = fullyConnectedLayer(5, 'Name', 'fc5');
    if isfield(wData, 'fc_weights') && isfield(wData, 'fc_bias')
        newFC.Weights = single(wData.fc_weights);
        newFC.Bias = single(wData.fc_bias);
        fprintf('  -> Injected calibrated hybrid weights into fc5 layer.\n');
    end

    lgraph = replaceLayer(lgraph, 'fc1000', newFC);
    lgraph = removeLayers(lgraph, 'ClassificationLayer_fc1000');

    fprintf('[3/3] Instantiating dlnetwork graph...\n');
    net = dlnetwork(lgraph);

    % Save compiled dlnetwork to .mat file for instant loading
    save(outputMat, 'net', '-v7.3');
    fprintf('\n[SUCCESS] Native MATLAB dlnetwork successfully saved to:\n  %s\n\n', outputMat);
end
