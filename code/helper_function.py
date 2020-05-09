import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import math
from moviepy.editor import VideoFileClip



# Calibration parameters------------------------------------------------------------------

# nx = 8 # the number of inside corners in x
# ny = 6 # the number of inside corners in y

# show_chess_corners=False

# undistort_an_example=True

#-----------------------------------------------------------------------------------------

# need a class later for these params




def calibrate( nx, ny, calib_folder_address, file_name_prefix,show_chess_corners=False,undistort_an_example=True): # calibrate the camera based on the given images

    # init object points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Read all the images from the folder
    images = glob.glob(calib_folder_address+file_name_prefix+'*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        
        img = cv2.imread(fname)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        if ret == True:

            # store the points
            objpoints.append(objp)
            
            imgpoints.append(corners)

            # Draw and display the corners
            if show_chess_corners:
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.imshow(img)
                cv2.imwrite('corners_drawn/'+fname.split("/")[-1],img)
                cv2.waitKey(500)
        else:
            print ("Shervin Error => Corners of the chessboard not found")

    img_shape=img.shape
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_shape[1], img_shape[0]), None, None)

    if undistort_an_example:
        dst = cv2.undistort(cv2.imread(images[0]), mtx, dist, None, mtx)
        cv2.imwrite('test_undist.jpg',dst)
        plt.imshow(dst)
        cv2.waitKey(1500)

    # Save the camera calibration data
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "../calibration_data/calib_mtx_dist_pickle.p", "wb" ) )


# undistort im using mtx and dist and return the result
def undistort(img):
    try:
        dist_pickle = pickle.load( open( "../calibration_data/calib_mtx_dist_pickle.p", "rb" ) )

        mtx = dist_pickle["mtx"]

        dist = dist_pickle["dist"]

    except:

        raise Exception("Shervin Error => No calibration data ")

    return cv2.undistort(img, mtx, dist, None, mtx)

def dir_threshold(img, sobel_kernel=3, angualr_thresh=(0, np.pi/2), color_threshold=(0, 255)):
    
    sobel_x,sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=sobel_kernel), cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=sobel_kernel)

    
    abs_x = np.absolute(sobel_x)
    
    abs_y = np.absolute(sobel_y)
    
    direction_g= np.arctan2(abs_y, abs_x)
    
    #scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    
    dir_sbinary = np.zeros_like(sobel_x)
    
    dir_sbinary[(direction_g >= angualr_thresh[0]) & (direction_g <= angualr_thresh[1])] = 1

    return dir_sbinary
    
    
def abs_sobel_thresh(img, orient='x',  thresh=(20,100), sobel_kernel=3, mono=False): # orinet x, y. suggested thress (20, 100)
    if orient=='x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in orient
    elif orient=='y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in orient
    else:
        raise Exception("orient parameter is not 'x' or 'y'")
    
    abs_sobel = np.absolute(sobel) # Absolute derivative
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Threshold x gradient
    s_binary = np.zeros_like(scaled_sobel)
    
    if mono:
        s_binary[(scaled_sobel[:,:,0] >= 20) & (scaled_sobel[:,:,0] <= 100) | (scaled_sobel[:,:,2] >= 20) & (scaled_sobel[:,:,2] <= 100) | (scaled_sobel[:,:,1] >= 20) & (scaled_sobel[:,:,1] <= 100)]= 1
    else:
        s_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])]= 1

    return s_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    sobel_x,sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=sobel_kernel), cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=sobel_kernel)

    magnitude = np.sqrt(sobel_x**2+sobel_y**2)
    
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    

    
    mag_binary = np.zeros_like(scaled_sobel)
    
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return mag_binary

def color_threshold(img, color_threshold=(90, 255), channel='s'): # channels: r g b h l s. suggested thresholds: r(200,255), s((90,255) or (170,255), h(15, 100)


    ind_dict={'r':0, 'g':1, 'b':2,'h':0, 'l':1, 's':2}

    if channel in ['r', 'g', 'b']:
        
        chl = img[:,:,ind_dict[channel]]
    
    elif channel in ['h', 'l', 's']:
        
        chl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,ind_dict[channel]]

    else:
        raise Exception("Channel unknown: "+channel)

    
    col_binary = np.zeros_like(img)
    
    col_binary[(chl > color_threshold[0]) & (chl <= color_threshold[1])] = 1

    return col_binary

def pipeline(img_distorted, s_thresh=(170, 255), sx_thresh=(20, 100)):
    
    img=undistort(img_distorted)

    # img = np.copy(img)
    
    col_bin_s=color_threshold(img, s_thresh, 's')

    col_bin_r=color_threshold(img, s_thresh, 'r')
    
    mag_bin=mag_thresh(img, 3, (30, 100))
    
    dir_bin=dir_threshold(img,15, (0.7, 1.3))
    
    sobel_x_bin=abs_sobel_thresh(img,'x', (20,100))
    
    sobel_y_bin=abs_sobel_thresh(img,'y')
    
    
    #print (col_bin.shape, sobel_x_bin.shape, sobel_y_bin.shape, mag_bin.shape,dir_bin.shape)
    
    combined = np.zeros_like(img)
    
    #ret[(col_bin_s==1) &((dir_bin==1) |(mag_bin ==1)) | ((sobel_x_bin == 1)& (sobel_y_bin == 1) & (col_bin_r==1))] = 1
    
    combined[((col_bin_r==1) &(col_bin_s==1)) | ((sobel_x_bin==1)&(mag_bin==1)) | ((dir_bin==1)&(sobel_x_bin==1))] = 1

    #ret[((col_bin_r==1) &(col_bin_s==1)) | ((sobel_x_bin==1)&(mag_bin==1)) | ((dir_bin==1)&(sobel_x_bin==1))] = 1   kheili khubea
    #print (ret.shape,ret)
    
    #print (col_bin.shape, sobel_x_bin.shape)

    #combined*=255

    combined_binary=np.zeros(combined.shape[:2])

    combined_binary[(combined[:,:,0]==1) | (combined[:,:,1]==1)|(combined[:,:,2]==1)]=1


    warped,M,lines=warp(combined_binary)

    #uncomment to see if the trapezoidal is OK
    # line_image_blank = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    # out_img = np.asarray(np.dstack((warped, warped, warped)), np.float64)
    # draw_lines(line_image_blank, lines, color=[255, 0, 0], thickness=10)
    # line_image_blank=np.asarray(line_image_blank, np.float64)
    # print out_img.shape, line_image_blank.shape
    # lines_overlayed_image=cv2.addWeighted(out_img, 0.8, line_image_blank, 1., 0.)
    # return lines_overlayed_image
    
    left_fitx, right_fitx, ploty=fit_polynomial(warped)

    res,center=warp_back(warped,M,left_fitx, right_fitx, ploty)

    # print(res.shape,img_distorted.shape)
    cur_x,curv_y=measure_curvature_pixels(res.shape)

    return_image=shadow_mix(img_distorted,res)

    font = cv2.FONT_HERSHEY_SIMPLEX

    org = (50, 50)

    fontScale = 1
   
    # Blue color in BGR 
    color = (255, 255, 0) 
      
    # Line thickness of 2 px 
    thickness = 2

    curv_x_text='Radius of Curvature: '+ str(round(cur_x,2))+ " Meters" if cur_x<3000 else "Strait Lines"

    cv2.putText(return_image, curv_x_text+ ", Center Offset: "+str(round(center*3.7/700,2)), org, font,  fontScale, color, thickness, cv2.LINE_AA) 

    cv2.putText(return_image,"Shervin Ghasemlou", (500,700), font,  0.8, (255,0,0), 3, cv2.LINE_AA)
    return return_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def get_warp_params(img):

    global i
    #original
    src_corners=[[[595,450],[690,450],[200,img.shape[0]],[1140,img.shape[0]]],

    #short, not working for all
    [[449,550],[857,550],[200,img.shape[0]],[1140,img.shape[0]]],

    # another
    [[552,480],[740,480],[200,img.shape[0]],[1140,img.shape[0]]]][i]

    dst_x1=340

    dst_x2=940

    dst_corners=[[dst_x1 ,0],[dst_x2,0],[dst_x1,img.shape[0]],[dst_x2,img.shape[0]]]

    src = np.float32(src_corners)
    
    dst = np.float32(dst_corners)

    lines = [[[dst_x1 ,0, dst_x1,img.shape[0]],[dst_x2,0,dst_x2,img.shape[0]]]]

    return src, dst, lines
    

def warp(img, src=None, dst=None):
    
    if not (src and dst):
        src, dst, lines = get_warp_params(img)

    M = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(img, M, img.shape[:2][::-1], flags=cv2.INTER_LINEAR) 


    
    return warped, M,lines

def warp_back(warped,m,left_fitx, right_fitx, ploty):

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    center=abs(np.average(pts_right[-1][0][0]+pts_left[-1][0][0])//2-warped.shape[1]//2)
    print(pts_right)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    invertible,m_inv=cv2.invert(m)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (color_warp.shape[1], color_warp.shape[0])) 

    return newwarp,center


def find_lane_pixels(binary_warped, draw_windows=False):
    #binary_warped/=255
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # plt.plot(histogram)

    # plt.show()
    # Create an output image to draw on and visualize the result
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        if draw_windows:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),   (win_xleft_high,win_y_high),(0,255,0), 2) 

            cv2.rectangle(out_img,(win_xright_low,win_y_low),   (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        
        if (len(good_left_inds)>minpix):
            leftx_current=int(np.mean(nonzerox[good_left_inds]))
        if (len(good_right_inds)>minpix):
            rightx_current=int(np.mean(nonzerox[good_right_inds]))    
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        #pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("Shervin Error => Some problem with either left or right lane indices")

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def search_around_poly(binary_warped,history_size=12):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_fit_gloabl_avg=np.average(left_fit_gloabl[-history_size:],axis=0)
    right_fit_gloabl_avg=np.average(right_fit_gloabl[-history_size:],axis=0)
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = (np.absolute(np.polyval(left_fit_gloabl_avg,nonzeroy)-nonzerox)<margin).nonzero()[0]

    right_lane_inds = (np.absolute(np.polyval(right_fit_gloabl_avg,nonzeroy)-nonzerox)<margin).nonzero()[0]
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped, history_size=12):
    # Find our lane pixels first
    global left_fit_gloabl, right_fit_gloabl, window_searched
    if window_searched:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped) 
    else:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
        window_searched=True
    

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit_gloabl_prefit = np.polyfit(lefty,leftx,2)# prefit because we want to check curvature before updating the global polyfit
    right_fit_gloabl_prefit = np.polyfit(righty,rightx,2)

    # print (np.sign(left_fit_gloabl_prefit[0]),np.sign(right_fit_gloabl_prefit[0]))

    if np.sign(left_fit_gloabl_prefit[0])== np.sign(right_fit_gloabl_prefit[0]):
        left_fit_gloabl.append(left_fit_gloabl_prefit)
        right_fit_gloabl.append(right_fit_gloabl_prefit)
    else:
        window_searched=True


    left_fit_gloabl_avg=np.average(left_fit_gloabl[-history_size:], axis=0)
    right_fit_gloabl_avg=np.average(right_fit_gloabl[-history_size:], axis=0)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit_gloabl_avg[0]*ploty**2 + left_fit_gloabl_avg[1]*ploty + left_fit_gloabl_avg[2]
        right_fitx = right_fit_gloabl_avg[0]*ploty**2 + right_fit_gloabl_avg[1]*ploty + right_fit_gloabl_avg[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty


    return left_fitx, right_fitx, ploty


def measure_curvature_pixels(image_shape,ym_per_pix = 30.0/720,xm_per_pix = 3.7/700): # meters per pixel in y dimension , # meters per pixel in x dimension
     
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image

    global left_fit_gloabl, right_fit_gloabl, window_searched

    left_fit_gloabl_avg_5=np.average(left_fit_gloabl[-2:],axis=0)
    right_fit_gloabl_avg_5=np.average(right_fit_gloabl[-2:],axis=0)   
    
    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)

    left_fitx = left_fit_gloabl_avg_5[0]*ploty**2 + left_fit_gloabl_avg_5[1]*ploty + left_fit_gloabl_avg_5[2]
    right_fitx = right_fit_gloabl_avg_5[0]*ploty**2 + right_fit_gloabl_avg_5[1]*ploty + right_fit_gloabl_avg_5[2]
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    Al,Bl=left_fit_gloabl_avg_5[:2]
    Ar,Br=right_fit_gloabl_avg_5[:2]
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
   
    return left_curverad, right_curverad

def shadow_mix(img1, img2, alpha=1, beta=0.3, gamma=0):
    
    # Combine the result with the original image
    result = cv2.addWeighted(img1, alpha, img2, beta, gamma)

    return result


def image_test():
    
    img=mpimg.imread("../test_images/straight_lines2.jpg")

    ret=pipeline(img)
    
    plt.imshow(ret)
    
    plt.show()
    
    cv2.waitKey(500)
    
    plt.close()




def video_test():
    global i
    j=[0,0,2]
    k=0
    for vid in ['project_video', 'challenge_video', 'harder_challenge_video']:#'project_video', 'challenge_video', 'harder_challenge_video'
        clip1 = VideoFileClip('../'+vid+'.mp4')
        
        white_clip = clip1.fl_image(process_image)
        
        white_clip.write_videofile(vid+'_processed.mp4', audio=False)
        i=j[k]
        k+=1

def process_image(img):
    return pipeline(img)

if __name__ == '__main__':
    
    window_searched=False
    i=0
    left_fit_gloabl = []
    right_fit_gloabl = []
    window_searched =False
    # image_test()
    video_test()