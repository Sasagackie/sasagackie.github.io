%% (1) Loading the dataset
% 1-1. Set a folder path to the dataset images 
imageFolder = '/Users/Katsuhirosasagaki/Downloads/PlantVillageDataset/train';

% 1-2. Load the entire dataset; a labeled name for images depends on their parent folder name
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
% tbl = countEachLabel(imds); % To see the count for each class

% To match the number of images in each class to the lowest number of images in the class.
    %minSetCount = min(tbl{:,2}); 
    %maxNumImages = minSetCount;
    %minSetCount = min(maxNumImages,minSetCount);


minSetCount = 1000; % Set 1000 for each class
imds = splitEachLabel(imds, minSetCount, 'randomize');

% countEachLabel(imds); To see the count for each class
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomize'); % Split those up in a 80:20 ratio

%% (2) Loading the CNN
% Load CNN
net = resnet50();

% Check and set the input size on the 1st layer
net.Layers(1)
inputSize = net.Layers(1).InputSize;

% Find the final classification & learnable layers
lgraph = layerGraph(net); % Convert the pretrained network into the layer graph
[learnableLayer,classLayer] = findLayersToReplace(lgraph); % Define these layers

% Replace the classification & learnable layers
numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


%% (3) Image Augmentations & Training Option Settings 
% Image augmentation Configurations
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Training Options
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% (4) Performing Training
net = trainNetwork(augimdsTrain,lgraph,options);

%% (5) Showing Results
[YPred,~] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);
% Confusion Matrix
YValidation = imdsValidation.Labels
confMat = confusionmat(YValidation, YPred)
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
mean(diag(confMat))

%% (6) Classify validation images that are not used for the training

% Change the image folder to the valid folder
imageFolder = '/Users/Katsuhirosasagaki/Downloads/PlantVillageDataset/valid'

% Set the datastore
imdsValidation = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% Change the input size
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Classify
[YPred,probs] = classify(net,augimdsValidation);

% Claculate Accuracy
accuracy = mean(YPred == imdsValidation.Labels)

% Confusion Matrix
YValidation = imdsValidation.Labels
confMat = confusionmat(YValidation, YPred);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
mean(diag(confMat))

% Export the ConfMat to a csv file
writematrix(ConfMat, 'filename.csv');

%% (7) Classify a single image
% cd 38Classes/
% load('ResnetTrained9950Percent.mat');
[filename, pathname] = uigetfile('*','Pick a leaf Image' );
testImage = imread([pathname,filename]);
[ParentFolderPath]=fileparts(pathname);
[~, ParentFolderName] = fileparts(ParentFolderPath);
testLabel = ParentFolderName

augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImage);
predictedLabel = classify(net,augimdsValidation)
