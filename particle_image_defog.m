%image defogging for pariticle imgaes captured in foggy environment. 
function output=particle_img_defog(img,step_length,threshold,winsize)
buchang=step_length;
yuzhi=1-threshold;
aaa=1;
ccc=1;
lameda=0.00001; %平滑参数
img=uint8(double(img));
img1=double(img);
[chang kuan]=size(img);
img7=dark_mid(img,5);
an1=dark_an(img7,10);
an2=dark_an(img7,winsize);
% %% 暗通道平滑
n=2;
W2 = 1./max( (abs(double(an2)/255-double(img7)/255)).^n,0.001)/1000;   % weight map
out2=mat2gray(W2);
q = wls_optimization(double(an2),out2, double(img),lameda); 
%% A
Amin=max(double(img(:)));
imwuqing=double(img1-q);
imav=mean(imwuqing(:));
A=Amin;
for A=Amin:buchang:2500
t=1-(q./A);   
j2=(double(img)-A.*(1-t))./(t.^2);
j1=double(uint8((double(img)-A.*(1-t))./(t.^2)));
j11(:,:,1)=uint8(j1);
j11(:,:,2)=uint8(j1);
j11(:,:,3)=uint8(j1);
j2=max(j1(:));
yiny(1,aaa)=A;
yiny(3,aaa)=mean(double(j1(:)));
anj1=dark_an(uint8(j1),5);
yiny(8,aaa)=max(anj1(:));
j11(:,:,1)=j1;
j11(:,:,2)=j1;
j11(:,:,3)=j1;
if aaa>1
   yiny(4,aaa)=(yiny(3,aaa-1)-yiny(3,aaa))./buchang; 
   yiny(7,aaa-1)=((yiny(3,aaa-1)-imav)./(yiny(3,1)-imav)); 
  if yiny(7,aaa-1)<yuzhi
      yiny(7,1)=1;    
      break;
   end
   youbutong=1;
   im =double(j1);
   if size(im,3)==1
   I(1:chang,1:kuan,1:3,ccc)  = cat(3,im,im,im);    
   ccc=ccc+1;
   end
end
aaa=aaa+1;
end
imgs_rgb=[];
%% 浓雾图像融合
if youbutong==1
N=ccc-1;

means = zeros(N,1);
for n = 1 : N
    means(n) = mean(mean(mean(I(:,:,:,n))));
end
[~, idx] = sort(means);
I_sort = zeros(chang,kuan,3,N);
for n = 1 : N
    imgs_rgb(:,:,:,n) = I(:,:,:,idx(n));
end
imgs_rgb = imgs_rgb/255.0;
w1 = get_weight11(imgs_lum);
w2 =1;
p1 = 1;
p2 = 1;
w = (w1.^p1).*(w2.^p2);
w = refine_weight(w);
lev = 5;
img_result = fusion_pyramid(imgs_rgb, w, lev);
output=uint8(255.*img_result(:,:,1)); 
end
end

%% 子函数
function output=dark_an(img,r)
if ndims(img) == 3
    dc = rgb2gray(img);
else
    dc = img;
end
%***********最小值滤波****************************、、
r_win=r;
last=r_win*r_win;
middle=last/2;
dc_an=ordfilt2(dc,1,ones(r_win,r_win),'symmetric');
output=dc_an;
end

function weight = get_weight11(img_seq_lum)
[H,W,N] = size(img_seq_lum);
weight = zeros(H,W,N);
% compute mean value of non-exposed intensity region
means = mean(mean(img_seq_lum));
means = reshape(means, N,1);
[meansav,PS]=mapminmax(means',0,1);
%compute sigma value of non-exposed intensity region
sigmas = zeros(N,1);
sigmas(:,1) = 0.3;
for n = 1 : N
    weight(:,:,n) =(exp(-0.5*(img_seq_lum(:,:,n) - abs(1-meansav(n))).^2/sigmas(n)/sigmas(n))); %   imshow([weight(:,:,n),img_seq_lum(:,:,n)])
end
a=1;
end
function out = wls_optimization(in, data_weight, guidance, lambda)
%Weighted Least Squares optimization solver.
% Given an input image IN, we seek a new image OUT, which, on the one hand,
% is as close as possible to IN, and, at the same time, is as smooth as
% possible everywhere, except across significant gradients in the hazy image.
%
%  Input arguments:
%  ----------------
%  in             - Input image (2-D, double, N-by-M matrix).   
%  data_weight    - High values indicate it is accurate, small values
%                   indicate it's not.
%  guidance       - Source image for the affinity matrix. Same dimensions
%                   as the input image IN. Default: log(IN)
%  lambda         - Balances between the data term and the smoothness
%                   term. Increasing lambda will produce smoother images.
%                   Default value is 0.05 
%
% This function is based on the implementation of the WLS Filer by Farbman,
% Fattal, Lischinski and Szeliski, "Edge-Preserving Decompositions for 
% Multi-Scale Tone and Detail Manipulation", ACM Transactions on Graphics, 2008
% The original function can be downloaded from: 
% http://www.cs.huji.ac.il/~danix/epd/wlsFilter.m
if size(guidance,3) == 1
    guidance = repmat(guidance,[1,1,3]);
end
small_num = 0.00001;
if ~exist('lambda','var') || isempty(lambda), lambda = 0.05; end
[h,w,~] = size(guidance);
k = h*w;
guidance = rgb2gray(guidance);
% Compute affinities between adjacent pixels based on gradients of guidance
jump=1;
dy = diff(guidance, 1, 1);  dy(dy>jump)=1;
dy = -lambda./(sum(abs(dy).^2,3) + small_num);
dy = padarray(dy, [1 0], 'post');
dy = dy(:);
dx = diff(guidance, 1, 2);  dx(dx>jump)=1;
dx = -lambda./(sum(abs(dx).^2,3) + small_num);
dx = padarray(dx, [0 1], 'post');
dx = dx(:);
% Construct a five-point spatially inhomogeneous Laplacian matrix
B = [dx, dy];
d = [-h,-1];
tmp = spdiags(B,d,k,k);
ea = dx;
we = padarray(dx, h, 'pre'); we = we(1:end-h);
so = dy;
no = padarray(dy, 1, 'pre'); no = no(1:end-1);
D = -(ea+we+so+no);
Asmoothness = tmp + tmp' + spdiags(D, 0, k, k);
% Normalize data weight
data_weight = data_weight+small_num;
Adata = spdiags(data_weight(:), 0, k, k);
A = Adata + Asmoothness;
b = Adata*in(:);
out = A\b;
out = reshape(out, h, w);
end
function output=dark_mid(img,r)
if ndims(img) == 3
    dc = rgb2gray(img);
else
    dc = img;
end
%***********最小值滤波****************************、、
r_win=r;
last=r_win*r_win;
middle=(last+1)/2;
dc_mid=ordfilt2(dc,middle,ones(r_win,r_win),'symmetric');
output=dc_mid;
end

function weight_out = refine_weight(weight_in)

[H,W,N] = size(weight_in);
weight_out = zeros(H,W,N);

for n = 1 : N
    weight_out(:,:,n) = imgaussfilt(weight_in(:,:,n), 5);
end
end

function result = fusion_pyramid(imgs, weights, lev)
[H,W,C,N] = size(imgs);
if(~exist('lev', 'var')),
    lev = floor(log(min(H,W)) / log(2));
end
% normalize weight
weights = weights + 1e-12; %avoids division by zero
weights = weights./repmat(sum(weights,3),[1 1 N]);
% create empty pyramid
pyr = gaussian_pyramid(zeros(H,W,C),lev);
% nlev = length(pyr);
% multiresolution blending
for n = 1 : N
    % construct pyramid from each input image
	pyrW = gaussian_pyramid(weights(:,:,n),lev);
    pyrI = laplacian_pyramid(imgs(:,:,:,n),lev);
    for l = 1:lev
        w = repmat(pyrW{l},[1 1 C]);
        pyr{l} = pyr{l} + w.*pyrI{l};
    end
end                                     
end

