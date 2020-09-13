cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
debuggingTest = load('.\Project1DataFiles\debuggingTest.mat');
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');

for classindex = 1:10
    %get indices of all images of that class
    inds = find(cifar10.trueclass==classindex);
    %take first one
    imrgb = cifar10.imageset(:,:,:,inds(1));
    %display it along with ground truth text label
    figure; imagesc(imrgb); 
    title(sprintf('\%s',cifar10.classlabels{classindex}));
end

for d = 1:length(CNNparameters.layertypes)
    fprintf('layer %d is of type %s\n',d,CNNparameters.layertypes{d});
    filterbank = CNNparameters.filterbanks{d};
    if not(isempty(filterbank))
        fprintf('filterbank size %d x %d x %d x %d\n', ...
        size(filterbank,1),size(filterbank,2), ...
        size(filterbank,3),size(filterbank,4));
        biasvec = CNNparameters.biasvectors{d};
        fprintf('number of biases is %d\n',length(biasvec));
    end
end

%sample code to show image and access expected results
figure; imagesc(imrgb); truesize(gcf,[64 64]);
for d = 1:length(debuggingTest.layerResults)
result = debuggingTest.layerResults{d};
fprintf('layer %d output is size %d x %d x %d\n',...
d,size(result,1),size(result,2), size(result,3));
end
%find most probable class
classprobvec = squeeze(debuggingTest.layerResults{end});
[maxprob,maxclass] = max(classprobvec);
%note, classlabels is defined in ’cifar10testdata.mat’
fprintf('estimated class is %s with probability %.4f\n',...
classlabels{maxclass},maxprob);

function outarray = apply_imnormalize(inarray)
    outarray = (inarray/255.0) - 0.5;
end

function outarray = apply_relu(inarray)
    outarray = max(inarray,0);
end

function outarray = apply_maxpool(inarray)
    % assuming all inarray has even number of row and column
    % get row, column and channel
    [n,m,d] = size(inarray);
    %make a stub outarray for now, we will give it real value later
    outarray = ones(n/2,m/2,d);
    for k = 1:d
        for i = 1:2:n
            for j = 1:2:m
            outarray((i+1)/2,((j+1)/2),k) = max(max(inarray(i:i+1, j:j+1)));
            end
        end
    end
end
% outarray = apply_convolve(inarray, filterbank, biasvals)
% 
% inarray is NxMxD1, filterbank is RxCxD1xD2,
% biasvals is a length D2 vector, and outarray is NxMxD2
% outarray = apply_fullconnect(inarray, filterbank, biasvals)
% inarray is NxMxD1, filterbank is NxMxD1xD2,
% biasvals is a length D2 vector, and outarray is 1x1xD2
% outarray = apply_softmax(inarray)
% inarray is 1x1xD and outarray is the same size
