%% This is a demo of 2D color image segmentation

%   Copyright by Quan Wang, 2012/12/16
%   Please cite: Quan Wang. GMM-Based Hidden Markov Random Field for 
%   Color Image and 3D Volume Segmentation. arXiv:1212.4527 [cs.CV], 2012.
    
clear;clc;close all;

mex BoundMirrorExpand.cpp;
mex BoundMirrorShrink.cpp;

for n = 1:10
    path = "../../../Data/";
    write_path = "../../../HMRF Results/";
    image = strcat("scene",string(n),"/",string(n),"_SWIR_hist");
    I=imread(strcat(path,image,".jpeg"));
    Y=double(I);
    Y(:,:,1)=gaussianBlur(Y(:,:,1),3);
    Y(:,:,2)=gaussianBlur(Y(:,:,2),3);
    Y(:,:,3)=gaussianBlur(Y(:,:,3),3);
    
    for k = 5:8
        %k=4; % k: number of regions
        g=6; % g: number of GMM components
        beta=1; % beta: unitary vs. pairwise
        EM_iter=5; % max num of iterations
        MAP_iter=10; % max num of iterations
        
        tic;
        try
            fprintf('Performing k-means segmentation\n');
            [X, GMM]=image_kmeans(Y,k,g);
            imwrite(uint8(X*80),'initial labels.png');
            
            [X, GMM]=HMRF_EM(X,Y,GMM,k,g,EM_iter,MAP_iter,beta);
            imwrite(uint8(X*80),strcat(write_path,image,"_hmrf_k=",string(k),".png"));
        catch 
            continue
        end
        toc;
    end
end