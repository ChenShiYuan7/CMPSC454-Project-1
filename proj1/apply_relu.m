function outarray = apply_relu(inarray)
    inarray = double(inarray);
    outarray = max(inarray,0);
end