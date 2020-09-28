cifar10 = load('.\Project1DataFiles\cifar10testdata.mat');
debuggingTest = load('.\Project1DataFiles\debuggingTest.mat');
CNNparameters = load('.\Project1DataFiles\CNNparameters.mat');
outarray = apply_fullconnect(debuggingTest.layerResults{16}, CNNparameters.filterbanks{17}, CNNparameters.biasvectors{17});

x = round(outarray,10)
y = round(debuggingTest.layerResults{17},10)
fprintf('layer %d is %d\n',17, isequal( x, y ));


outarraySoft = apply_softmax(x);
fprintf('layer %d is %d\n',18, isequal( outarraySoft, debuggingTest.layerResults{18} ));
compareresult =  debuggingTest.layerResults{18};
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

        
        
        
        
