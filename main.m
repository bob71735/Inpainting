imgName = '456';
maskName = '456_mask.png';
psz1 = 15; psz2 = 31; 
[inpaintedLabel,inpaintedEdge,inpaintedD,fillRegion,DL,fillMovieL,intersum] = label_inpainting(imgName,maskName,psz1);
[inpaintedD(:,:,1) inpaintedD(:,:,2) inpaintedD(:,:,3)] = deal(inpaintedD);
[inpaintedDepth,C,DD,fillMovieD] = inpainting(inpaintedD,fillRegion,psz2)