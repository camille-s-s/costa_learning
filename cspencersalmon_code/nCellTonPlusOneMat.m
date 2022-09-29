function m = nCellTonPlusOneMat(c)

if ~iscell(c)
    disp('Input must be a cell. ')
    return
end

if size(unique(cell2mat(cellfun(@size, c, 'un', 0)), 'rows'), 1) ~= 1
    disp('Dimensions of each entry in input must be identical. ')
    return
end

nDim = ndims(c{1});
otherDim = repmat({':'}, 1, nDim);

m = zeros([size(c{1}), length(c)]); % newest added dimension is same as length of input

for n = 1 : length(c)
    m(otherDim{:}, n) = c{n};
end
    

end