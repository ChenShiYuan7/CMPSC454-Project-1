cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
debuggingTest = load('.\Project1DataFiles\debuggingTest.mat');
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');

fprintf('This demo routine will display a few images to show intermediate result,\n side by side is the intermediate result from debuggingTest.mat\n');
fprintf('It will also print the probability of identifying object with our routine \n and debug result');
fprintf('Finally, it shows a bar graph of the final softmax layer.\n');

% demo code: runs our routines on the debugging image, take no input
figure('name','original input image'); imagesc(debuggingTest.imrgb); truesize(gcf,[64 64]); 

inarray = debuggingTest.imrgb;
outarray = apply_imnormalize (inarray);
layer = 1;
title = strcat('Layer ',num2str(layer));
title = strcat(title, ' result, after normalize');
figure('name',title); imagesc(outarray);


% do layer 2 to 16 in a loop, with filter bank and vias starting at 2
for repeat = 1:3
    
    layer = layer + 1;
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    title = strcat('Layer ',num2str(layer));
    title = strcat(title, ' result, after Convolve');
    figure('name',title); 
    for i = 1:size(outarray,3)
       subplot(3,4,i), imshow(outarray(:,:,i)); 
    end
    layer = layer + 1;
    outarray = apply_relu(outarray);
    
    layer = layer + 1;
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    
    layer = layer + 1;
    outarray = apply_relu(outarray);
    
    layer = layer + 1;
    outarray = apply_maxpool(outarray);
    title = strcat('Layer ',num2str(layer));
    title = strcat(title, ' result, after Maxpool');
    figure('name',title); 
    for i = 1:size(outarray,3)
       subplot(3,4,i), imshow(outarray(:,:,i)); 
    end
end

outarray = apply_fullconnect(outarray, CNNparameters.filterbanks{17}, CNNparameters.biasvectors{17});

outarray = apply_softmax(outarray);
bargraph = [];
for i = 1:size(outarray,3)
    bargraph(i) = outarray(1,1,i); 
end
bargraphLabel = categorical(cifar10.classlabels);
figure('name','Softmax Result');bar(bargraphLabel,bargraph);

%print result
classprobvec = squeeze(debuggingTest.layerResults{end});
[maxprob,maxclass] = max(classprobvec);
fprintf('\n estimated class with debugging result is %s with probability %.4f\n',...
cifar10.classlabels{maxclass},maxprob);

classprobvec = squeeze(outarray);
[maxprob,maxclass] = max(classprobvec);
fprintf('estimated class with our routine is %s with probability %.4f\n',...
cifar10.classlabels{maxclass},maxprob);
