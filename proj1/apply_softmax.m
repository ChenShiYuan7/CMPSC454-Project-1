
function outarray = apply_softmax(inarray)
    inarray = double(inarray);
%   find alpha, which is just max element
    alpha = max(inarray);
    denominator = 0;
    outarray = zeros(size(inarray));
%   calculate denominator part
    for i = 1:size(inarray,3)
        temp = exp(inarray(:,:,i) - alpha);
        denominator = denominator + temp;
    end
    
%   calculate numerator part and get softmax for each element
    for i = 1:size(inarray,3)
        numerator = exp(inarray(:,:,i) - alpha);
        outarray(:,:,i) = numerator/denominator;
    end
end