cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
debuggingTest = load('.\Project1DataFiles\debuggingTest.mat');
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');

% demo code: runs our routines on the debugging image, take no input
figure('name', 'original picture'); imagesc(debuggingTest.imrgb); truesize(gcf,[64 64]);

inarray = debuggingTest.imrgb;
outarray = apply_imnormalize (inarray);
layer = 1;

% do layer 2 to 16 in a loop, with filter bank and vias starting at 2
for repeat = 1:3
    
    layer = layer + 1;
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    outarray = round(outarray,10);
    
    layer = layer + 1;
    outarray = apply_relu(outarray);
    
    layer = layer + 1;
    outarray = apply_convolve(outarray, CNNparameters.filterbanks{layer}, CNNparameters.biasvectors{layer});
    outarray = round(outarray,10);
    
    layer = layer + 1;
    outarray = apply_relu(outarray);
    outarray = round(outarray,10);
    
    layer = layer + 1;
    outarray = apply_maxpool(outarray);
    outarray = round(outarray,10);
end

outarray = apply_fullconnect(outarray, CNNparameters.filterbanks{17}, CNNparameters.biasvectors{17});

outarray = apply_softmax(outarray);


%print result
classprobvec = squeeze(debuggingTest.layerResults{end});
[maxprob,maxclass] = max(classprobvec);
fprintf('estimated class with debugging result is %s with probability %.4f\n',...
cifar10.classlabels{maxclass},maxprob);

classprobvec = squeeze(outarray);
[maxprob,maxclass] = max(classprobvec);
fprintf('estimated class with our routine is %s with probability %.4f\n',...
cifar10.classlabels{maxclass},maxprob);
