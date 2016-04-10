function [inpaintedLabel,inpaintedEdge,inpaintedDepth,fillRegion,D,fillMovie,intersum] = label_inpainting(imgName,maskName,psz)

imgFileName = strcat(imgName,'_label.png');
depthName = strcat(imgName,'_depth.png');
argu_num = 7; endure = 100; range = 0;
mask = imread(maskName);
Img = double(imread(imgFileName));
Depth = double(imread(depthName));
[Gmag Gdir] = imgradient(rgb2gray(Img));
Gdir = double(Gdir > 0);

fillRegion = mask(:,:,1) == 255;
[Gdir(:,:,1) Gdir(:,:,2) Gdir(:,:,3)] = deal(Gdir);
[Depth(:,:,1) Depth(:,:,2) Depth(:,:,3)] = deal(Depth);
imgdir = Gdir;
label = Img;
depth = Depth;
ind = img2ind(imgdir);
sz = [size(imgdir,1) size(imgdir,2)];
sourceRegion = ~fillRegion;

% Initialize isophote values
[Ix Iy] = gradient(rgb2gray(imgdir));
[Ix Iy] = deal(-Iy/255,Ix/255);

% Initialize confidence and data terms
D = repmat(-.1,sz);
iter = 1;
% Visualization stuff
if nargout==argu_num
  fillMovie(1).cdata=uint8(label); 
  fillMovie(1).colormap=[];
  Img(1,1,:) = [0, 255, 0];
  iter = 2;
end

% Seed 'rand' for reproducible results (good for testing)
rand('state',0);

% Loop until entire fill region has been covered
while any(fillRegion(:))
  % Find contour & normalized gradients of fill region 
  dR = find(conv2(double(fillRegion),[1,1,1;1,-8,1;1,1,1],'same') > 0);
  
  [Nx,Ny] = gradient(double(~fillRegion));
  N = [Nx(dR(:)) Ny(dR(:))];
  N = normr(N);  
  N(~isfinite(N))=0;
  
  inproduct = abs(Ix(dR).*N(:,1)+Iy(dR).*N(:,2));
  if(~inproduct)
      break;
  end
  D(dR) = inproduct;% + 0.001;
  
  % Compute confidences along the fill front
  %{
  for k=dR'
    Hp = getpatch(sz,k,psz);
    q = Hp(~(fillRegion(Hp)));   
    C(k) = sum(C(q))/numel(Hp);
  end
  %}
  % Compute patch priorities = confidence term * data term
  priorities = D(dR);
  
  % Find patch with maximum priority, Hp
  [~,ndx] = max(priorities(:));
  p = dR(ndx(1));
  [Hp,rows,cols] = getpatch(sz,p,psz);
  toFill = fillRegion(Hp);
  r1 = max(1,(min(rows)-range)); r2 = min(sz(1),(max(rows) + range)); 
  c1 = max(1,(min(cols)-range)); c2 = min(sz(2),(max(cols) + range)); 
  
  % Find exemplar that minimizes error, Hq
  I = imgdir;
  %figure;imshow(I);
  Hq = bestexemplar(I,imgdir(rows,cols,:),toFill',sourceRegion);%,r1,c1);
  
  % Update fill region
  toFill = logical(toFill);   
  fillRegion(Hp(toFill)) = false;
  
  % Propagate confidence & isophote values
  Ix(Hp(toFill)) = Ix(Hq(toFill));
  Iy(Hp(toFill)) = Iy(Hq(toFill));
  
  % Copy image data from Hq to Hp
  ind(Hp(toFill)) = ind(Hq(toFill));
  
  imgdir(rows,cols,:) = ind2img(ind(rows,cols),Gdir);
  label(rows,cols,:) = ind2img(ind(rows,cols),Img);
  depth(rows,cols,:) = ind2img(ind(rows,cols),Depth);
  %figure;imshow(imgdir);
  % Visualization stuff
  if nargout==argu_num
    ind2 = ind;
    ind2(logical(fillRegion)) = 1;
    fillMovie(iter).cdata = uint8(ind2img(ind2,Img)); 
    fillMovie(iter).colormap = [];
  end
  iter = iter + 1;
end
imgdir = rgb2gray(imgdir);
imgdir(find(fillRegion)) = 0;
inpaintedEdge = imgdir;

depth = depth(:,:,1);
depth(find(fillRegion)) = 0;
inpaintedDepth = uint8(depth);

label = rgb2gray(uint8(label));
[B,L] = bwboundaries(fillRegion,'noholes');
%imshow(label2rgb(L, @jet, [0 0 0]))
c = unique(label);
intersum = zeros(size(B,1),size(c,1));

for i = 1:size(B,1)
	for j = 1:size(c,1)
        label(find(L == i)) = c(j);
        [nx ny] = gradient(double(label));
        n = nx.^2 + ny.^2;
        inter = intersect(find(L == i),find(n > 0),'rows');
        intersum(i,j) = size(inter,1);
        if intersum(i,j) < endure;  
            %fillRegion(find(L == i)) = 0;
            L(find(L == i)) = 0;
            break;
        end
	end
end

inpaintedLabel = label;

%---------------------------------------------------------------------
% Scans over the entire image (with a sliding window)
% for the exemplar with the lowest error. Calls a MEX function.
%---------------------------------------------------------------------
function Hq = bestexemplar(img,Ip,toFill,sourceRegion)%,r1,c1)
m = size(Ip,1); mm = size(img,1); n = size(Ip,2); nn = size(img,2);
best = bestexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion);
%figure;imshow(img(best(1):best(2),best(3):best(4),:))
%best(1) = best(1) + r1 - 1; best(2) = best(2) + r1 -1;
%best(3) = best(3) + c1 - 1; best(4) = best(4) + c1 -1;
Hq = sub2ndx(best(1):best(2),(best(3):best(4))',mm);
%---------------------------------------------------------------------
% Returns the indices for a 9x9 patch centered at pixel p.
%---------------------------------------------------------------------
function [Hp,rows,cols] = getpatch(sz,p,psz)
% [x,y] = ind2sub(sz,p);  % 2*w+1 == the patch size
w = (psz-1)/2; p = p-1; y = floor(p/sz(1))+1; p = rem(p,sz(1)); x = floor(p)+1;
rows = max(x-w,1):min(x+w,sz(1));
cols = (max(y-w,1):min(y+w,sz(2)))';
Hp = sub2ndx(rows,cols,sz(1));

%---------------------------------------------------------------------
% Converts the (rows,cols) subscript-style indices to Matlab index-style
% indices.  Unforunately, 'sub2ind' cannot be used for this.
%---------------------------------------------------------------------
function N = sub2ndx(rows,cols,nTotalRows)
X = rows(ones(length(cols),1),:);
Y = cols(:,ones(1,length(rows)));
N = X+(Y-1)*nTotalRows;


%---------------------------------------------------------------------
% Converts an indexed image into an RGB image, using 'img' as a colormap
%---------------------------------------------------------------------
function img2 = ind2img(ind,img)
for i=3:-1:1, temp=img(:,:,i); img2(:,:,i)=temp(ind); end

%---------------------------------------------------------------------
% Converts an RGB image into a indexed image, using the image itself as
% the colormap.
%---------------------------------------------------------------------
function ind = img2ind(img)
s=size(img); ind=reshape(1:s(1)*s(2),s(1),s(2));

