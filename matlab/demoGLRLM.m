clc;
clear;

A = [5, 2, 5, 4, 4; 
         3, 3, 3, 1, 3;
         2, 1, 1, 1, 3; 
         4, 2, 2, 2, 3;
         3, 5, 3, 3, 2]
mask = ones(size(A(:, :)));
stats = regionprops(mask, 'BoundingBox');
bx = int16(floor(stats.BoundingBox)) + int16(floor(stats.BoundingBox)==0)
     