clc;clear all;close all;
load('svm_classifier.mat');
%colorImage = imread('handicapSign.jpg');
colorImage = imread('streetsign2.jpg');
I = rgb2gray(colorImage);
%I = histeq(I);

% Detect MSER regions.
[mserRegions, mserConnComp] = detectMSERFeatures(I, ...
    'RegionAreaRange',[200 8000],'ThresholdDelta',4);

figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('MSER regions')
hold off

%% SECOND PART:

% Use regionprops to measure MSER properties
mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
    'Solidity', 'Extent', 'Euler', 'Image');

% Compute the aspect ratio using bounding box data.
bbox = vertcat(mserStats.BoundingBox);
w = bbox(:,3);
h = bbox(:,4);
aspectRatio = w./h;

% Threshold the data to determine which regions to remove. These thresholds
% may need to be tuned for other images.
filterIdx = aspectRatio' > 3;
filterIdx = filterIdx | [mserStats.Eccentricity] > .995 ;
filterIdx = filterIdx | [mserStats.Solidity] < .3;
filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
filterIdx = filterIdx | [mserStats.EulerNumber] < -8;

% Remove regions
mserStats(filterIdx) = [];
mserRegions(filterIdx) = [];

% Show remaining regions
figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Geometric Properties')
hold off

%% THIRD PART:

% Get a binary image of the a region, and pad it to avoid boundary effects
% during the stroke width computation.
regionImage = mserStats(6).Image;
regionImage = padarray(regionImage, [1 1]);

% Compute the stroke width image.
distanceImage = bwdist(~regionImage);
figure; imshow(distanceImage);
skeletonImage = bwmorph(regionImage, 'thin', inf);
figure; imshow(skeletonImage);

strokeWidthImage = distanceImage;
strokeWidthImage(~skeletonImage) = 0;
figure; imshow(strokeWidthImage);

% Show the region image alongside the stroke width image.
figure
subplot(1,2,1)
imagesc(regionImage)
title('Region Image')

subplot(1,2,2)
imagesc(strokeWidthImage)
title('Stroke Width Image')

% Compute the stroke width variation metric
strokeWidthValues = distanceImage(skeletonImage);
strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

% Threshold the stroke width variation metric
strokeWidthThreshold = 0.4;
strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;

% Process the remaining regions
for j = 1:numel(mserStats)

    regionImage = mserStats(j).Image;
    regionImage = padarray(regionImage, [1 1], 0);

    distanceImage = bwdist(~regionImage);
    skeletonImage = bwmorph(regionImage, 'thin', inf);

    strokeWidthValues = distanceImage(skeletonImage);

    strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

    strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;

end

% Remove regions based on the stroke width variation
mserRegions(strokeWidthFilterIdx) = [];
mserStats(strokeWidthFilterIdx) = [];

% Show remaining regions
figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Stroke Width Variation')
hold off


%% FOURTH PART:

% Get bounding boxes for all the regions
bboxes = vertcat(mserStats.BoundingBox);

bboxes = sortrows(bboxes,[1 2]);

% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax] format for convenience.
xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

% Expand the bounding boxes by a small amount.
% expansionAmount = 0.02;
% xmin = (1-expansionAmount) * xmin;
% ymin = (1-expansionAmount) * ymin;
% xmax = (1+expansionAmount) * xmax;
% ymax = (1+expansionAmount) * ymax;

% Clip the bounding boxes to be within the image bounds
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));

% Show the expanded bounding boxes
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
expandedBBoxes = ceil(expandedBBoxes);

for i=1:size(expandedBBoxes,1)
    characterImages{i}=imresize(I(expandedBBoxes(i,2):(expandedBBoxes(i,2)+expandedBBoxes(i,4)-1) , expandedBBoxes(i,1):(expandedBBoxes(i,1)+expandedBBoxes(i,3)-1)), [28 28]);
end

IExpandedBBoxes = insertShape(colorImage,'Rectangle',expandedBBoxes,'LineWidth',3);

figure
imshow(IExpandedBBoxes)
title('Expanded Bounding Boxes Text')

%% Recognize characters:

cellSize = [4 4];
for i = 1:size(characterImages,2)
    characterFeatures(i, :) = extractHOGFeatures(characterImages{i}, 'CellSize', cellSize);
end

% % Make class predictions using the test features.
predictedLabels = predict(classifier, characterFeatures);

%%

characters={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};

finalstring=[];
for i=1:size(predictedLabels,1)
    finalstring=[finalstring, characters(predictedLabels(i))]
end
%finalstring=strcat(finalstring)
% 
% % Compute the overlap ratio
% overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);
% 
% % Set the overlap ratio between a bounding box and itself to zero to
% % simplify the graph representation.
% n = size(overlapRatio,1);
% overlapRatio(1:n+1:n^2) = 0;
% 
% % Create the graph
% g = graph(overlapRatio);
% 
% % Find the connected text regions within the graph
% componentIndices = conncomp(g);
% 
% % Merge the boxes based on the minimum and maximum dimensions.
% xmin = accumarray(componentIndices', xmin, [], @min);
% ymin = accumarray(componentIndices', ymin, [], @min);
% xmax = accumarray(componentIndices', xmax, [], @max);
% ymax = accumarray(componentIndices', ymax, [], @max);
% 
% % Compose the merged bounding boxes using the [x y width height] format.
% textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
% 
% % Remove bounding boxes that only contain one text region
% numRegionsInGroup = histcounts(componentIndices);
% textBBoxes(numRegionsInGroup == 1, :) = [];
% 
% % Show the final text detection result.
% ITextRegion = insertShape(colorImage, 'Rectangle', textBBoxes,'LineWidth',3);
% 
% figure
% imshow(ITextRegion)
% title('Detected Text')
% 
% ocrtxt = ocr(I, textBBoxes);
% [ocrtxt.Text]