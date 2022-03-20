%image defogging for pariticle imgaes captured in foggy environment. 
clear all
fileFolder=fullfile('D:\A_Image\haze\lab\fog3\3\example\');
dirOutput=dir(fullfile(fileFolder,'*.jpg'));
fileNames={dirOutput.name}';
[k len]=size(fileNames);
step_length=40; 
threshold=0.95;
winsize=35;

for i=1:k
name=fileNames{i,1};
name1=strcat(fileFolder,name);
image=imread(name1);
img=rgb2gray(image);
result=particle_img_defog(img,step_length,threshold,winsize);
imshow([img, result])
end
