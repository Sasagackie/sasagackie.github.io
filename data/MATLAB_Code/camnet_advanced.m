function value = camnet_advanced(cam,nnet)
    % Camera settings
    cam.Autofocus = 'on'; % Autofocus
    img = snapshot(cam,'manual'); % Capturing mode
    pic = imresize(img,[224,224]); % Image size
    imshow(pic)
    % A Leaf Segmentation
    L = superpixels(pic,500); % Generate a label matrix
    f = drawrectangle(gca,'Position',[92 92 40 40],'Color','g'); % Generate a rectanguler ROI
    foreground = createMask(f,pic); % Make a foreground mask
    % Generate Rectanguler Non-ROIs for each corner of the image
    b1 = drawrectangle(gca,'Position',[5 200 10 10],'Color','r');
    b2 = drawrectangle(gca, 'Position',[5 15 10 10], 'Color', 'r');
    b3 = drawrectangle(gca, 'Position',[200 15 10 10], 'Color', 'r');
    b4 = drawrectangle(gca, 'Position',[200 200 10 10], 'Color', 'r');
    % Generate a union of those
    background = createMask(b1,pic) + createMask(b2,pic) + createMask(b3,pic) + createMask(b4,pic);
    BW = lazysnapping(pic,L,foreground,background); % Run a lazysnapping
    % Show the extracted leaf
    imshow(labeloverlay(pic,BW,'Colormap',[0 1 0]))
    maskedImage = pic;
    maskedImage(repmat(~BW,[1 1 3])) = 0;
    imshow(maskedImage)
    % Classify the extracted leaf
    value = classify(nnet,maskedImage);
    image(maskedImage)
    title(char(value)) % Show the result
end
