function [lArrayW, lArrayV] = indexIdentify_Array8(Neurons)
% This function is used toidentify index of array, hemispere and moneky for
% each neuron in eight-Utah Arrary experment.
% Written by Hua Tang,Postdoc of Dr. Averbeck Lab, @NIMH on July 13, 2019.
% last updated: July 13, 2019

% if monkey w:
% A = L_rd, F = L_md, B = L_cd, E = L_v
% H = R_rd, G = R_md, C = R_cd, D = R_v
iArrayW = {'A', 'F', 'B', 'E'; ...
    'H', 'G', 'C', 'D'}; 

% if monkey v:
% B = L_rd, F = L_md, A = L_cd, E = L_v
% G = R_rd, C = R_md, D = R_cd, H = R_v
iArrayV = {'B', 'F', 'A', 'E'; ...
    'G', 'C', 'D', 'H'};

% #units x 3 where columns in order are 1) monkey (1 or 2 for w or v), 2)
% "hemi" aka row ix for matching array, and 3) "array" aka col ix for
% matching array
iNeuron = []; 

for m = 1 : size(Neurons, 1)
    if strcmpi(Neurons{m, 1}, 'w')
        iArray = iArrayW; % array layout for w
        iNeuron(m, 1) = 1;
    elseif strcmpi(Neurons{m, 1}, 'v')
        iArray = iArrayV; % array layout for v
        iNeuron(m, 1) = 2;
    end
    
    [hemi, array]=find(strcmp(Neurons{m, 2}, iArray));
    iNeuron(m, 2) = hemi;
    iNeuron(m, 3) = array;
end

% create neuron list for different array, hemispere and moneky
for i = 1 : 4 % array (column)
    for j = 1 : 2 % hemisphere (row)
        lArrayW{j, i} = find(iNeuron(:, 1) == 1 & iNeuron(:, 2) == j & iNeuron(:, 3) == i);
        lArrayV{j, i} = find(iNeuron(:, 1) == 2 & iNeuron(:, 2) == j & iNeuron(:, 3) == i);
    end
end