cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
debuggingTest = load('.\Project1DataFiles\debuggingTest.mat');
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');

% test code:
figure; imagesc(debuggingTest.imrgb); truesize(gcf,[64 64]);
% result = debuggingTest.layerResults{d};
% fprintf('layer %d output is size %d x %d x %d\n',...
% d,size(result,1),size(result,2), size(result,3));
inarray = debuggingTest.imrgb;
outarray = apply_imnormalize (inarray);
layer = 1;
result1 = debuggingTest.layerResults{layer};
isequal( outarray, result1)
% do layer 2 to 16 in a loop, with filter bank and vias starting at 2
filterLayer = 2;
for repeat = 1:3
    
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{filterLayer}, CNNparameters.biasvectors{filterLayer});
    filterLayer = filterLayer + 2;
    result1 = debuggingTest.layerResults{layer};
    isequal( outarray, result1)
    layer = layer + 1;
    
    outarray = apply_relu(outarray);
    result1 = debuggingTest.layerResults{layer};
    isequal( outarray, result1)
    layer = layer + 1;
    
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{filterLayer}, CNNparameters.biasvectors{filterLayer});
    filterLayer = filterLayer + 2;
    result1 = debuggingTest.layerResults{layer};
    isequal( outarray, result1)
    layer = layer + 1;
    
    outarray = apply_relu(outarray);
    result1 = debuggingTest.layerResults{layer};
    isequal( outarray, result1)
    layer = layer + 1;
    
    outarray = apply_maxpool(outarray);
    result1 = debuggingTest.layerResults{layer};
    isequal( outarray, result1)
    layer = layer + 1;
end
% outarray = apply_fullconnect(outarray, filterbank, biasvals);
% outarray = apply_softmax(outarray, filterbank, biasvals);


% for classindex = 1:10
%     %get indices of all images of that class
%     inds = find(cifar10.trueclass==classindex);
%     %take first one
%     imrgb = cifar10.imageset(:,:,:,inds(1));
%     %display it along with ground truth text label
%     figure; imagesc(imrgb); truesize(gcf,[64 64]);
%     title(sprintf('\%s',cifar10.classlabels{classindex}));
% end

% for d = 1:length(CNNparameters.layertypes)
%     fprintf('layer %d is of type %s\n',d,CNNparameters.layertypes{d});
%     filterbank = CNNparameters.filterbanks{d};
%     if not(isempty(filterbank))
%         fprintf('filterbank size %d x %d x %d x %d\n', ...
%         size(filterbank,1),size(filterbank,2), ...
%         size(filterbank,3),size(filterbank,4));
%         biasvec = CNNparameters.biasvectors{d};
%         fprintf('number of biases is %d\n',length(biasvec));
%     end
% end

%sample code to show image and access expected results
% figure; imagesc(imrgb); truesize(gcf,[64 64]);
% for d = 1:length(debuggingTest.layerResults)
% result = debuggingTest.layerResults{d};
% fprintf('layer %d output is size %d x %d x %d\n',...
% d,size(result,1),size(result,2), size(result,3));
% end
%find most probable class
% classprobvec = squeeze(debuggingTest.layerResults{end});
% [maxprob,maxclass] = max(classprobvec);
% %note, classlabels is defined in ’cifar10testdata.mat’
% fprintf('estimated class is %s with probability %.4f\n',...
% cifar10.classlabels{maxclass},maxprob);

function outarray = apply_imnormalize(inarray)
    inarray = double(inarray);
    outarray = (inarray/255.0) - 0.5;
end

function outarray = apply_relu(inarray)
    inarray = double(inarray);
    outarray = max(inarray,0);
end

function outarray = apply_maxpool(inarray)
    % assuming all inarray has even number of row and column
    % get row, column and channel
    inarray = double(inarray);
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

function outarray = apply_convolve(inarray, filterbank, biasvals)
    % get size of filterbank and inarray, create outarray
    inarray = double(inarray);
    [R,C,D1,D2] = size(filterbank);
    [N,M,D] = size(inarray);
    outarray = ones(N,M,D2);
    % do convolution D2 times, which D2 is how many filter we have in the
    % fiterbank
    for filterNumber = 1:D2
        outarray(:,:,filterNumber) = imfilter(inarray, filterbank(:,:,:,filterNumber),'same','conv',0);
        outarray(:,:,filterNumber) = outarray(:,:,filterNumber) + biasvals(filterNumber);
    end
      
end

function outarray = apply_fullconnect(inarray, filterbank, biasvals)

end

function outarray = apply_softmax(inarray)
    inarray = double(inarray);
%   find alpha
    alpha = max(inarray);
    sum = 0;
    
%   calculate denominator part
    for i = 1:len(inarray)
        a = exp(inarray(:,:,i)) - alpha;
        sum = sum + a;
    end
    
%   calculate numerator part and get softmax for each k
    for i = 1:len(inarray)
        b = exp(inarray(:,:,i)) - alpha;
        inarray(:,:,i) = b/sum;
    end
    outarray = inarray;
end
