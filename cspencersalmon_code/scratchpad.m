% X = [10 162 3 119 73 2; 1 1 1 7 4 671];

X = [70 65 17 1 5 4; 0 11 0 12 4 25];

classEntropy = @(classCount, sizeClust) -1*((classCount/sizeClust) * log2(classCount/sizeClust));
[nClust, nClass] = size(X);
clustEntropy = NaN(nClust, 1);
clustPurity = NaN(nClust, 1);

for clust = 1 : nClust
    Xi = X(clust, :);
    sizeClusti = sum(Xi);
    
    clustEntropy(clust) = sum(arrayfun(@(classlbl) classEntropy(Xi(classlbl), sizeClusti), 1:nClass));
    clustPurity(clust) = max(Xi) ./ sizeClusti;
end

