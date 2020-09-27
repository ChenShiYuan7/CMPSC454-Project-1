cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
debuggingTest = load('.\Project1DataFiles\debuggingTest.mat');
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');
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

% test code:
figure; imagesc(debuggingTest.imrgb); truesize(gcf,[64 64]);
% result = debuggingTest.layerResults{d};
% fprintf('layer %d output is size %d x %d x %d\n',...
% d,size(result,1),size(result,2), size(result,3));
inarray = debuggingTest.imrgb;
outarray = apply_imnormalize (inarray);
layer = 1;
result1 = debuggingTest.layerResults{1,1};
fprintf('layer %d is %d\n',layer, isequal( outarray, debuggingTest.layerResults{layer}));
% do layer 2 to 16 in a loop, with filter bank and vias starting at 2
for repeat = 1:3
    
    layer = layer + 1;
    outarray = apply_convolve(debuggingTest.layerResults{layer-1}, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    outarray = round(outarray,10);
    fprintf('layer %d is %d\n',layer, isequal( outarray, round(debuggingTest.layerResults{layer},10)));
%     fprintf('layer %d is %d\n',layer, isequal( outarray, debuggingTest.layerResults{layer}));
    
    layer = layer + 1;
    outarray = apply_relu(outarray);
    fprintf('layer %d is %d\n',layer, isequal( outarray, round(debuggingTest.layerResults{layer},10)));
    
    layer = layer + 1;
    outarray = apply_convolve(debuggingTest.layerResults{layer-1}, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    outarray = round(outarray,10);
    fprintf('layer %d is %d\n',layer, isequal( outarray, round(debuggingTest.layerResults{layer},10)));
    
    layer = layer + 1;
    outarray = apply_relu(outarray);
    outarray = round(outarray,10);
    fprintf('layer %d is %d\n',layer, isequal( outarray, round(debuggingTest.layerResults{layer},10)));
    
    layer = layer + 1;
    outarray = apply_maxpool(debuggingTest.layerResults{layer-1});
    outarray = round(outarray,10);
    fprintf('layer %d is %d\n',layer, isequal( outarray, round(debuggingTest.layerResults{layer},10)));
    compare = round(debuggingTest.layerResults{layer},10);
end
outarray = apply_fullconnect(debuggingTest.layerResults{16}, CNNparameters.filterbanks{17}, CNNparameters.biasvectors{17});
fprintf('layer 17 is %d\n', isequal( outarray, debuggingTest.layerResults{17}));
outarray = apply_softmax(debuggingTest.layerResults{17});
fprintf('layer 18 is %d\n', isequal( outarray, debuggingTest.layerResults{18}));
% end of test code

% for classindex = 1:10
%     get indices of all images of that class
%     inds = find(cifar10.trueclass==classindex);
%     take first one
%     imrgb = cifar10.imageset(:,:,:,inds(1));
%     display it along with ground truth text label
%     figure; imagesc(imrgb); truesize(gcf,[64 64]);
%     title(sprintf('\%s',cifar10.classlabels{classindex}));
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
    outarray = double(outarray);
    for k = 1:d
        for i = 1:2:n
            for j = 1:2:m
            outarray((i+1)/2,((j+1)/2),k) = max(max(inarray(i:i+1, j:j+1, k)));
            end
        end
    end
end

function outarray = apply_convolve(inarray, filterbank, biasvals)
    % get size of filterbank and inarray, create outarray
    if (isempty(filterbank))
        fprintf('error, no filter bank');
        return
    end
    [R,C,D1,D2] = size(filterbank);
    [N,M,D] = size(inarray); %D1 and D should be equal
    if D1 ~= D
        fprintf('error, Dimension not equal D1:%d D:%d\n',D1,D);
        return
    end
    outarray = double(zeros(N,M,D2));
    % do convolution D2 times, which D2 is how many filter we have in the
    % fiterbank
    
    for filterNumber = 1:size(filterbank,4)
        filter = filterbank(:,:,:,filterNumber);
        sumOfAllChannel = zeros(N,M);
        for channel = 1:size(filterbank,3)
            temp = (imfilter(inarray(:,:,channel),filter(:,:,channel),0, 'conv','same'));
            sumOfAllChannel = sumOfAllChannel + temp;
        end
        outarray(:,:,filterNumber) = double(sumOfAllChannel + double(biasvals(filterNumber)));
       
        if size(outarray,1) ~= N
            fprintf('error, Dimension not equal N:%d outarray N:%d\n',N,size(outarray,1));
        end
        if size(outarray,2) ~= M
            fprintf('error, Dimension not equal M:%d outarray M:%d\n',M,size(outarray,2));
        end
    end
      outarray = double(outarray);
end

function outarray = apply_fullconnect(inarray, filterbank, biasvals)
    
  outSize = [1, 1, 1];
  outSize(3) = size(filterbank, 4);
  outarray = zeros(outSize);
 
  for l = 1:size(filterbank, 4)
    for i = 1:size(inarray, 1)
      for j = 1:size(inarray, 2)
        for k = 1:size(inarray, 3)
          outarray(1, 1, l) = outarray(1, 1, l) + inarray(i, j, k) * filterbank(i, j, k, l);
        end
      end
    end
    outarray(1, 1, l) = outarray(1, 1, l) + biasvals(l);   
  end

end

function outarray = apply_softmax(inarray)
    inarray = double(inarray);
%   find alpha
    alpha = max(inarray);
    sum = 0;
    
%   calculate denominator part
    for i = 1:length(inarray)
        a = exp(inarray(:,:,i)) - alpha;
        sum = sum + a;
    end
    
%   calculate numerator part and get softmax for each k
    for i = 1:length(inarray)
        b = exp(inarray(:,:,i)) - alpha;
        inarray(:,:,i) = b/sum;
    end
    outarray = inarray;
end
