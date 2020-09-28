%this file generates the confusion matrix, and calculates the accuracy of
%our CNN, and generates Top k=1 graph. Takes more than 20 mins to run on my
%computer 
cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
ConfusionMatrix = zeros(10,10);
top1KaccuracybyClass = zeros(1,10);
top2KaccuracybyClass = zeros(1,10);
top3KaccuracybyClass = zeros(1,10);
for classindex = 1:10
    % go through each class's pictures and try to make a confusion matrix
    inds = find(cifar10.trueclass==classindex);
    % do 1000 image of each class
    for image = 1:1000
        imrgb = cifar10.imageset(:,:,:,inds(image));
        outarray = apply_cnn(imrgb);
        % apply CNN, get softmax result, and use provided method to find the most likely class
        classprobvec = squeeze(outarray);
        [maxprob,maxclass] = max(classprobvec);
        % save result to Confusion Matrix
        ConfusionMatrix(classindex,maxclass) = ConfusionMatrix(classindex,maxclass) + 1;
        % compute if ground truth class is in the top 2 or 3 score,
        % topKCheck return 1 if it does , 0 otherwise so we just add
        top2KaccuracybyClass(classindex) = top2KaccuracybyClass(classindex) + topKCheck(outarray, 3, classindex);
        top3KaccuracybyClass(classindex) = top3KaccuracybyClass(classindex) + topKCheck(outarray, 2, classindex);
    end
end

%caculate accuracy using provided formula in pdf
numerator = 0;
denominator = sum(ConfusionMatrix, 'all');

for groundTruthClass = 1:10
    numerator = numerator + ConfusionMatrix(groundTruthClass,groundTruthClass);
    top1KaccuracybyClass(groundTruthClass) = ConfusionMatrix(groundTruthClass,groundTruthClass) / sum(ConfusionMatrix(groundTruthClass,:));
end
% this is accuracy of simply diagonal values over sum of all
accuracy = double(numerator) / double (denominator);
fprintf('Accuracy of our CNN program is %d\n', accuracy);
disp(ConfusionMatrix);
figure('name','K=1 ClassificationAccuracy Chart');bar(categorical(cifar10.classlabels), top1KaccuracybyClass);
figure('name','K=2 ClassificationAccuracy Chart');bar(categorical(cifar10.classlabels), top2KaccuracybyClass);
figure('name','K=3 ClassificationAccuracy Chart');bar(categorical(cifar10.classlabels), top3KaccuracybyClass);

function outarray = apply_cnn(inarray)
% do layer 1
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');
outarray = apply_imnormalize (inarray);
layer = 1;
% do layer 2 to 16 in a loop
for repeat = 1:3
    
    layer = layer + 1;
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    
    layer = layer + 1;
    outarray = apply_relu(outarray);
    
    layer = layer + 1;
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    
    layer = layer + 1;
    outarray = apply_relu(outarray);
   
    layer = layer + 1;
    outarray = apply_maxpool(outarray);
end
% do last 2 layer
outarray = apply_fullconnect(outarray, CNNparameters.filterbanks{17}, CNNparameters.biasvectors{17});

outarray = apply_softmax(outarray);
end

function truthClassInProbVec = topKCheck(probVec, topk, truthClass)
% tells if the truthClass index is in the top K probability score
       truthClassInProbVec = 0;
       temp = sort(probVec,'descend');
       topvalue = zeros(size(probVec,2));
       topClasses = zeros(size(probVec,2));
       for k = 1:topk
       topvalue(k) = temp(k);
       topClasses(k) = find((probVec==topvalue(k)));
       end
       groundTruthClassInTopK = find(topClasses==truthClass);
       if isempty(groundTruthClassInTopK)
           % if ground truth class not found in topK return false
           truthClassInProbVec = 0;
       else
           truthClassInProbVec = 1;
       end
    
end