clc
clear
% Normalization
Im = imread('ROCtraining/image23_training.jpg');

Im_R = Im(:,:,1);   
Im_G = Im(:,:,2);   
Im_B = Im(:,:,3);
[H,W] = size(Im_R); 

I_new_R = adapthisteq(Im_R);
I_new_G = adapthisteq(Im_G);
I_new_B = adapthisteq(Im_B);

I_new = im2uint8(zeros(H,W,3));
I_new(:,:,1) = I_new_R;
I_new(:,:,2) = I_new_G;
I_new(:,:,3) = I_new_B;
figure(1)
imshow(I_new)

% Morlet Wavelet in all directions

angz = 0:pi/180:pi;
Morlet_MA_mat = cellmat(1,length(angz),H,W);
Morlet_MA = cwtft2(I_new_R,...
    'wavelet',{'morl',{4,1,1}},...
    'scales',5,...
    'angles',angz);

Morlet_MA_min = 50;

% Select min pixel
for angn = 1:length(angz)
    Temp = abs(Morlet_MA.cfs(:,:,1,1,angn));
    Morlet_MA_min = min(Morlet_MA_min,Temp);
    Morlet_MA_mat{1,angn} = Temp;
end

figure (2)
imagesc(Morlet_MA_min)
colorbar;

MA_bin = imbinarize(Morlet_MA_min,7);
figure (3)
imshow(MA_bin);

% Mark MA point (suggested)
[B,L,N,A] = bwboundaries(MA_bin);
figure (4)
imshow(I_new); hold on;
colors=['b'];
for k=1:length(B)
  boundary = B{k};
  cidx = mod(k,length(colors))+1;
  plot(boundary(:,2), boundary(:,1),...
       colors(cidx),'LineWidth',2);
end
hold off
