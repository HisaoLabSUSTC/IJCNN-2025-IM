function res = NormalizeObjs(val, PF) % nomrmalize the obecjtive values by PF
    N = size(val,1);
    fmin   = min(PF, [], 1);
    fmax   = max(PF,[],1);
    res = (val-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
end 