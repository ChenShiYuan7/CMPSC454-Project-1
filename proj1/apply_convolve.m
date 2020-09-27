
function outarray = apply_convolve(inarray, filterbank, biasvals)
    % get size of filterbank and inarray, create outarray
    if (isempty(filterbank))
        fprintf('error, no filter bank');
        return
    end
    [R,C,D1,D2] = size(filterbank);
    [N,M,D] = size(inarray); %D1 and D should be equal
    if D1 ~= D
        fprintf('error, Dimension not equal D1:%d D:%d\n',D1,D);
        return
    end
    outarray = double(zeros(N,M,D2));
    % do convolution D2 times, which D2 is how many filter we have in the
    % fiterbank
    
    for filterNumber = 1:size(filterbank,4)
        filter = filterbank(:,:,:,filterNumber);
        sumOfAllChannel = zeros(N,M);
        for channel = 1:size(filterbank,3)
            temp = (imfilter(inarray(:,:,channel),filter(:,:,channel),0, 'conv','same'));
            sumOfAllChannel = sumOfAllChannel + temp;
        end
        outarray(:,:,filterNumber) = double(sumOfAllChannel + double(biasvals(filterNumber)));
       
        if size(outarray,1) ~= N
            fprintf('error, Dimension not equal N:%d outarray N:%d\n',N,size(outarray,1));
        end
        if size(outarray,2) ~= M
            fprintf('error, Dimension not equal M:%d outarray M:%d\n',M,size(outarray,2));
        end
    end
      outarray = double(outarray);
end
