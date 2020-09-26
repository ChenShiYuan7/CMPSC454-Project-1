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