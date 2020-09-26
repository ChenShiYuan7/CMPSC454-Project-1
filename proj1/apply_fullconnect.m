function outarray = apply_fullconnect(inarray, filterbank, biasvals)
    
    %transfer value to the double first
    inarray = double(inarray);
    outarray = zeros(1,1,10,'double');
    
    %make a 1*1*10 image matrix 
    for i = 1:10
        %calculate the forth dimension of filtetbank
        fl = filterbank(:,:,:,i);
        
        product = dot(inarray,fl);
        
        %get the sum of dot profuct for fl and input
        sums = sum(product(:));
        
        %get the value which include the bias 
        outarray(:,:,i) = sums + biasvals(i);
        
    end
    
    
    
 end
        
        
        
        
        
