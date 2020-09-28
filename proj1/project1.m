cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
debuggingTest = load('.\Project1DataFiles\debuggingTest.mat');
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');
for classindex = 1:10
    
    inds = find(cifar10.trueclass==classindex);

    imrgb = cifar10.imageset(:,:,:,inds(1));

    figure; imagesc(imrgb); truesize(gcf,[64 64]);
    title(sprintf('\%s',cifar10.classlabels{classindex}));
end

% test code: verfies each layer's correctness by comparing to debug result
figure; imagesc(debuggingTest.imrgb); truesize(gcf,[64 64]);

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


