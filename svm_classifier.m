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


cellSize = [4 4];
%hogFeatureSize = length(hog_4x4);

% The trainingSet is an array of 10 imageSet objects: one for each digit.
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

trainingFeatures = [];
trainingLabels=list.ALLlabels(list.TRNind(:,end),:);

for i = 1:size(trainingFiles,1)
    trainingFeatures(i, :) = extractHOGFeatures(xTrainImages{i}, 'CellSize', cellSize);
    %thefeatures=detectSURFFeatures(xTrainImages{i})
    %trainingFeatures(i,:) = thefeatures.Location;
end

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels);

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

for i = 1:size(testingFiles,1)
    testingFeatures(i, :) = extractHOGFeatures(xTestImages{i}, 'CellSize', cellSize);
    %thefeatures=detectSURFFeatures(xTestImages{i});
    %testingFeatures(i,:) = thefeatures.Location;
end

% % Make class predictions using the test features.
predictedLabels = predict(classifier, testingFeatures);
% 
% % Tabulate the results using a confusion matrix.
confMat = confusionmat(testingLabels, predictedLabels);

precision = diag(confMat)./sum(confMat,2);
precisionMean = mean(precision)