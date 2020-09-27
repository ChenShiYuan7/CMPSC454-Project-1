function outarray = apply_maxpool(inarray)
    % assuming all inarray has even number of row and column
    % get row, column and channel
    inarray = double(inarray);
    [n,m,d] = size(inarray);
    %make a stub outarray for now, we will give it real value later
    outarray = ones(n/2,m/2,d);
    outarray = double(outarray);
    for k = 1:d
        for i = 1:2:n
            for j = 1:2:m
            outarray((i+1)/2,((j+1)/2),k) = max(max(inarray(i:i+1, j:j+1, k)));
            end
        end
    end
end