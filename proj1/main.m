
cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
ConfusionMatrix = zeros(10,10);

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
    end
end

%caculate accuracy using provided formula in pdf
numerator = 0;
denominator = sum(ConfusionMatrix, 'all');
for groundTruthClass = 1:10
    numerator = numerator + ConfusionMatrix(groundTruthClass,groundTruthClass);
end
accuracy = double(numerator) / double (denominator);
fprintf('Accuracy of our program is %d\n', accuracy);
disp(ConfusionMatrix);


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

