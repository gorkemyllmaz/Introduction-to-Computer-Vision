% Author: Görkem Yilmaz
% Date: March 6, 2020
% Introduction to Computer Vision
% Homework 1

source_im = "3b.png";
source_image = imread(source_im);
if size(source_im) == 3 
    source_image = rgb2gray(source_image);
end

% Test part for Question 1
figure;
histo = histogram(source_image);
bar(histo);
xlabel('Intensity');
ylabel('Count');
title("Histogram for: " + source_im);

% To compare with MATLAB's built-in image histogram function
figure;
imhist(source_image);

% Test part for Question 2
binary_image = otsu_threshold(source_image);
figure("Name", source_im);
imshow(binary_image);
xlabel("Binary Image: " + source_im);

% Test part for Question 3
struct_el = ones(3, 3);


binary_image = otsu_threshold(source_image);
dilated_image = dilation(binary_image, struct_el);
figure("Name", source_im);
imshow(dilated_image);
xlabel("Dilated Image: " + source_im);


binary_image = otsu_threshold(source_image);
eroded_image = erosion(binary_image, struct_el);
figure("Name", source_im);
imshow(eroded_image);
xlabel("Eroded Image: " + source_im);


% Test part for Question 4

% inversion of the image for 3b.png
% binary_image = imcomplement(source_image);
binary_image = otsu_threshold(source_image); % if you uncomment imcomplement, then give binary_image as parameter here.

result_image = erosion(binary_image, ones(3,3));
result_image = dilation(result_image, ones(3,3));

figure("Name", source_im);
imshow(result_image);
xlabel("Dilated & Eroded Image: " + source_im);
title("Morphological Opeations: " + source_im);


% Labeling the binary image
bwlabeled_image = bwlabel(result_image);
% Making the labels rgb
component_label = label2rgb(bwlabeled_image);
figure;
imshow(component_label);
xlabel("Labeled Image: " + source_im);


% Implementation the histogram function for Question 1
function histo = histogram(source_image)
% E-mail states that some images are not grayscale.
    imdata = source_image;
    [rows,columns] = size(imdata); 
    histo = zeros(256,1);
    for row = 1:rows
      for column = 1:columns
        counter = double(imdata(row,column)) + 1;
        histo(counter) = histo(counter) + 1;    
      end
    end
end
% Implementation otsu's method for Question 2
function binary_image = otsu_threshold(source_image)
    imageMatrix = source_image;
    % create histogram of source image.
    [rows, columns] = size(imageMatrix);
    % sum all values.
    P = zeros(255);
    
    % Compute the histogram
    n = histogram(source_image);
    % Sum the values of all the histogram values
    N = sum(n);    
    
    for i=1:256
        % Compute the count rate of each intensity level.
        P(i)=n(i)/N;
    end
    
    % Initialize max and threshold as zero.
    max = 0;
    threshold = 0;
 
    for T = 2:255
        % Calculating background weight
        wb = sum(P(1:T)); 
        weightb = wb / (rows * columns);
        % Calculating background weight
        wf = sum(P(T+1:255));
        weightf = wf / (rows * columns);
        
        % Calculating background mean
        meanb = dot([1:T],P(1:T)) / wb;
        % Calculating foreground mean
        meanf = dot([T+1:255],P(T+1:255)) / wf;
        
        % Calculating between class variance (by the faster approach)
        variance = weightb * weightf * ((meanb - meanf)^2);
        
        % Threshold is where we find the maximum variance
        if (variance > max)
            max = variance;
            threshold = T;
        end
    end
    binary_image = zeros(size(imageMatrix));
    % Make it white where matrice value is greater than threshold
    binary_image(imageMatrix >= threshold) = 255;
    % Make it black where matrice value is less than threshold
    binary_image(imageMatrix < threshold) = 0;
end
% Implementation of dilation function for Question 3
function dilated_image = dilation(source_image, struct_el)
    % Getting the row & column count of structure element.
    [rows_el, columns_el] = size(struct_el);
    
    % Resulting dilated image: Contaions zeros for initalization.
    dilated_image = zeros(size(source_image, 1), size(source_image, 2));

    % Rows and columns of structure element should be the same size and
    % they should be odd
    if mod(rows_el, 2) == 0 || mod(columns_el, 2) == 0
        disp("Dimensions of structure element must be odd!");
        return;
    end
    
    % Itearate through source image to find any crossing between source
    % image and structuring element.
    for i = ceil(rows_el / 2): size(source_image, 1) - floor(rows_el / 2)
        for j = ceil(columns_el / 2): size(source_image, 2) - floor(columns_el / 2)
            % Create a temporary array with the size of structuring element. 
            temporary = source_image(i-floor(rows_el/2):i+floor(rows_el/2), j-floor(columns_el/2):j+floor(columns_el/2));  
            % See if temporary and structure element have any same value
            % at the same index.
            cross = temporary(logical(struct_el));
            dilated_image(i, j) = max(cross(:)); 
        end
    end
end
% Implementation of erosion function for Question 3
function eroded_image = erosion(source_image, struct_el)

    % Getting the row & column count of structure element.
    [rows_el, columns_el] = size(struct_el);
    
    % Resulting eroded image: Contaions zeros for initalization.
    eroded_image = zeros(size(source_image, 1), size(source_image, 2));
    
    % Rows and columns of structure element should be the same size and
    % they should be odd
    if mod(rows_el, 2) == 0 || mod(columns_el, 2) == 0
        disp("Dimensions of structure element must be odd!");
        return;
    end
    
    % Itearate through source image to find any crossing between source
    % image and structuring element.
    for i = ceil(rows_el / 2): size(source_image, 1) - floor(rows_el / 2)
        for j = ceil(columns_el / 2): size(source_image, 2) - floor(columns_el / 2)
            % Create a temporary array with the size of structuring element. 
            temporary = source_image(i-floor(rows_el/2):i+floor(rows_el/2), j-floor(columns_el/2):j+floor(columns_el/2));  
            
            % See if temporary has a different value in an element compared
            % to struct_el.
            cross = temporary(logical(struct_el));
            eroded_image(i, j) = min(cross(:));
            
        end
    end
end

