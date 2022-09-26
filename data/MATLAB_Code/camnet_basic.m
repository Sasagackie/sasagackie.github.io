function value = camnet_basic(cam,nnet)
    cam.Autofocus = 'on';
    img = snapshot(cam,'manual');
    pic = imresize(img,[224,224]);
    value = classify(nnet,pic);
    image(pic)
    title(char(value))
end