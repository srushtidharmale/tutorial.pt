import numpy as np
import cv2
import math
from playsound import playsound

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 150, 0], thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    #print('len(lines) ', lines)
    if lines is None:
        print('Lane Changed!')
        playsound('Lane.mp3')
        return image
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (float)(y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            #print('slope:', slope, 'line:', x1, y1, x2, y2)
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
                
    min_y = image.shape[0]/1.5 #*  (3 / 5) # <-- Just below the horizon
    max_y = image.shape[0] # <-- The bottom of the image
    
    min_y = int(min_y)
    #print('min_y:', min_y, 'max_y:', max_y, 'left_line_x', left_line_x)

    if len(left_line_x) == 0:
        return image

    if len(left_line_y) == 0:
        return image

    if len(right_line_x) == 0:
        return image

    if len(right_line_y) == 0:
        return image


    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))


    draw_lines(
        line_img,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
    )
        
    #cv2.imshow('Test image',line_img)
    #cv2.waitKey(500)
    
    draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
    )
    
    #draw_lines(line_img, lines)
    return image

def weighted_img(img, initial_img, a=0.8, b=1., y=0.):
    return cv2.addWeighted(initial_img, a, img, b, y)


#reading in an image
#image = cv2.imread('road1.jpg')
# open a pointer to the video stream and start the FPS timer
stream = cv2.VideoCapture('2.mp4')
print("Pune")
print("[18.5196, 73.8554]")

# loop over frames from the video file stream

while True:
	# grab the frame from the threaded video file stream
        (grabbed, image) = stream.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
        if not grabbed:
            break
	
        #cv2.imshow("Frame", image)
        #cv2.waitKey(1)

        height, width, channels = image.shape
        # Convert to grayscale here.
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 100, 200)

        region_of_interest_vertices = [
            (0, height),
            (width / 2, height / 2),
            (width, height),
        ]

        # Moved the cropping operation to the end of the pipeline.
        cropped_image = region_of_interest(
            cannyed_image,
            np.array([region_of_interest_vertices], np.int32)
        )


        #printing out some stats and plotting
        #print('This image is:', type(image), 'with dimensions:', image.shape)
        #cv2.imshow('Test image',image)
        #cv2.waitKey(100)
        #cv2.imshow('Test image',gray_image)
        #cv2.waitKey(100)
        #cv2.imshow('Test image',cannyed_image)
        #cv2.waitKey(100)
        #cv2.imshow('Test image',cropped_image)
        #cv2.waitKey(100)
        #print (cropped_image.shape)

        line_image = hough_lines(cropped_image, 6, (np.pi / 60), 160, 40, 80)
        #line_image = draw_lines(image, lines)
        cv2.imshow('Test image',line_image)
        cv2.waitKey(50)

        #cv2.waitKey(0)

cv2.destroyAllWindows()
