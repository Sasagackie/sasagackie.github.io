
[filename, pathname] = uigetfile('*','Pick a leaf Image' );
RGB = imread([pathname,filename]);
%[ParentFolderPath]=fileparts(pathname);
%[~, ParentFolderName] = fileparts(ParentFolderPath);
%testLabel = ParentFolderName
%fullFileName = fullfile(pathname, filename);
%baseFileName = filename;
imgWidth = 256;
RGB = imresize(RGB, [imgWidth imgWidth]);
imshow(RGB)
L = superpixels(RGB,500);
fWidth = 256/2.5;
fHeight = imgWidth/6.4;
f = drawrectangle(gca,'Position',[fWidth fWidth fHeight fHeight],'Color','g');
foreground = createMask(f,RGB);
b1 = drawrectangle(gca,'Position',[5 235 10 10],'Color','r');
b2 = drawrectangle(gca, 'Position',[5 15 10 10], 'Color', 'r');
b3 = drawrectangle(gca, 'Position',[235 15 10 10], 'Color', 'r');
b4 = drawrectangle(gca, 'Position',[235 235 10 10], 'Color', 'r');
background = createMask(b1,RGB) + createMask(b2,RGB) + createMask(b3,RGB) + createMask(b4,RGB);
BW = lazysnapping(RGB,L,foreground,background);
imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))
%imshow(labeloverlay(RGB,BW, 'Colormap','autumn','Transparency',0.25))
maskedImage = RGB;
maskedImage(repmat(~BW,[1 1 3])) = 0;
imshow(maskedImage)
