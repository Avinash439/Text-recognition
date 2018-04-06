clear all; clc; close all;
load('./Lists/English/Img/lists.mat');

trainingFiles=list.ALLnames(list.TRNind(:,end),:);

for i=1:size(trainingFiles,1)
    theimage=imread(['./English/Img/', trainingFiles(i,:),'.png']);
    if size(theimage,3)==3
        theimage=histeq(rgb2gray(theimage));
    end
    xTrainImages{i}=imbinarize(imresize(theimage,[28 28]));
    whiteindexes=find(xTrainImages{i}==1);
    blackindexes=find(xTrainImages{i}==0);
    if size(whiteindexes)<size(blackindexes)
        xTrainImages{i}=imcomplement(xTrainImages{i});
    end
end

trainingLabels=list.ALLlabels(list.TRNind(:,end),:);
tTrain=zeros(list.NUMclasses,size(trainingFiles,1));
for i=1:size(trainingFiles,1)
    tTrain(trainingLabels(i),i)=1;
end

%%
imshow(xTrainImages{100});

% Load the training data into memory
%[xTrainImages, tTrain] = digittrain_dataset;

%%

% Load the training data into memory
%[xTrainImages, tTrain] = digittrain_dataset;

% Display some of the training images
% clf
% for i = 1:20
%     subplot(4,5,i);
%     imshow(xTrainImages{i});
% end

rng('default')

hiddenSize1 = 500;

autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',300, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

plotWeights(autoenc1);

feat1 = encode(autoenc1,xTrainImages);

hiddenSize2 = 300;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',300, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc2)

feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);

view(softnet)

%%
deepnet = stack(autoenc1,autoenc2,softnet);

view(deepnet)

%%
% Get the number of pixels in each image
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Load the test images
%[xTestImages, tTest] = digittest_dataset;

testingFiles=list.ALLnames(list.TSTind(:,end),:);

for i=1:size(testingFiles,1)
    theimage=imread(['./English/Img/', testingFiles(i,:),'.png']);
    if size(theimage,3)==3
        theimage=histeq(rgb2gray(theimage));
    end
    xTestImages{i}=imbinarize(imresize(theimage,[28 28]));
    whiteindexes=find(xTestImages{i}==1);
    blackindexes=find(xTestImages{i}==0);
    if size(whiteindexes)<size(blackindexes)
        xTestImages{i}=imcomplement(xTestImages{i});
    end
end

testingLabels=list.ALLlabels(list.TSTind(:,end),:);
tTest=zeros(list.NUMclasses,size(testingFiles,1));
for i=1:size(testingFiles,1)
    tTest(testingLabels(i),i)=1;
end

%%

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end

y = deepnet(xTest);
%plotconfusion(tTest,y);

% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);

y = deepnet(xTest);

%%
[c,cm,ind,per]=confusion(tTest,y);
plotconfusion(tTest,y);