function outarray = softmax(in)
    
%   find alpha
    alpha = max(in);
    sum = 0;
    
%   calculate denominator part
    for i = 1:len(in)
        a = exp(in(:,:,i)) - alpha;
        sum = sum + a;
    end
    
%   calculate numerator part and get softmax for each k
    for i = 1:len(in)
        b = exp(in(:,:,i)) - alpha;
        in(:,:,i) = b/sum;
    end
    
    outarray = in;
        
    
    
end