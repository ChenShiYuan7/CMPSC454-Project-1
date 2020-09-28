% this matrix is generated from our main routine, we use it to generate the
% top k plot.
cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
ConfusionMatrix = [     531    41    65    37    10     8    18    38   210    42;
                        40   519     9    26    10     7    19    29   111   230;
                        87     8   386   117    97    70   104    88    25    18;
                        39    18   127   325    45   136   186    89    13    22;
                        53     6   270    69   259    38   162   114    22     7;
                        19     7   151   222    49   281   111   125    20    15;
                        10     7   120   125    93    23   557    33     9    23;
                        32     7    73    98    77    94    54   533    13    19;
                       192    84    35    44     7     8    10    16   542    62;
                        69   191    23    41     4     9    30    68   127   438;];

label = categorical(cifar10.classlabels);

K = 3;
plot = topKaccuracyPlot(ConfusionMatrix, K)
title = strcat('TopK= ',num2str(K));
title = strcat(title, ' Classification Accuracy Graph');
figure('name',title); bar(label,plot);

K = 1;
plot = topKaccuracyPlot(ConfusionMatrix, K)
title = strcat('TopK= ',num2str(K));
title = strcat(title, ' Classification Accuracy Graph');
figure('name',title); bar(label,plot);

K = 2;
plot = topKaccuracyPlot(ConfusionMatrix, K)
title = strcat('TopK= ',num2str(K));
title = strcat(title, ' Classification Accuracy Graph');
figure('name',title); bar(label,plot);

function outarray = topKaccuracyPlot(CM, topk)
% CM should be square matrix
    outarray = zeros(1,size(CM,1));
    for i = 1:size(CM,1)
        % get top k value and store them by finding topValue, and use
        % topValue to find top Classes
       temp = sort(CM(i,:),'descend');
       topvalue = zeros(size(CM(i,:)));
       topClasses = zeros(size(CM(i,:)));
       for k = 1:topk
       topvalue(k) = temp(k);
       topClasses(k) = find((CM(i,:)==topvalue(k)));
       end
       groundTruthClassInTopK = find(topClasses==i);
       if isempty(groundTruthClassInTopK)
           % if ground truth class not found in topK just say accuracy is 0
           outarray(i) = 0;
       else
           % else compute the accuracy depending on K
           for k = 1:topk
           outarray(i) = outarray(i) + CM(i,topClasses(k));
           end
           outarray(i) = outarray(i) / double(sum(CM(i,:)));
       end
    end
    
end