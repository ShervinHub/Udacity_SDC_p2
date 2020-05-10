import numpy as np
from collections import deque
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


class Lane():
    def __init__(self):
        self.left_line=Line()

        self.right_line=Line()

        self.window_searched=False

        self.center=None

        self.first_iteration=True

        self.lost=False

        self.image_class=0

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False 

        # x values of the last n fits of the line
        self.recent_xfitted = [] 

        #average x values of the fitted line over the last n iterations
        self.bestx = None    

        #polynomial coefficients  history
        self.recent_fits=deque(maxlen=4)

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  

        #radius of curvature of the line in some units
        self.radius_of_curvature = None 

        #distance in meters of vehicle center from the line
        self.line_base_pos = None 

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

        #x values for detected line pixels how
        self.allx = None  

        #y values for detected line pixels
        self.ally = None 

        # how does this differ from allx , all x is pre M_inv I think, confused a little myself
        self.pointx = None  

        #y values for detected line pixels
        self.pointy = None

        self.continuos_failures=0

        self.window_searched=False

def calibrate( nx, ny, calib_folder_address, file_name_prefix,show_chess_corners=False,undistort_an_example=True): # calibrate the camera based on the given images

    # init object points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Read all the images from the folder
    images = glob.glob(calib_folder_address+file_name_prefix+'*.jpg')

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

    elif channel=='v':
        chl = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2]
    else:
        raise Exception("Channel unknown: "+channel)

    
    col_binary = np.zeros_like(img)
    
    col_binary[(chl > color_threshold[0]) & (chl <= color_threshold[1])] = 1

    return col_binary

def pipeline(img_distorted, Lane, s_thresh=(170, 255), sx_thresh=(20, 100)):
    
    img=undistort(img_distorted)


    combined = np.zeros_like(img)

    if Lane.image_class==0:

        col_bin_s=color_threshold(img, s_thresh, 's')

        col_bin_r=color_threshold(img, s_thresh, 'r')

        combined[((col_bin_r==1) |(col_bin_s==1))] = 1

    elif Lane.image_class==1:

        col_bin_s=color_threshold(img, s_thresh, 's')

        col_bin_v=color_threshold(img, s_thresh, 'v')

        sobel_x_bin=abs_sobel_thresh(img,'x', (20,100))
        
        sobel_y_bin=abs_sobel_thresh(img,'y')

        combined[(sobel_x_bin == 1)& (sobel_y_bin == 1) | (col_bin_s == 1)& (col_bin_v == 1)]=1

    else:

        col_bin_r=color_threshold(img, s_thresh, 'r')

        col_bin_s=color_threshold(img, s_thresh, 's')

        col_bin_v=color_threshold(img, s_thresh, 'v')

        sobel_x_bin=abs_sobel_thresh(img,'x', (20,100))
        
        sobel_y_bin=abs_sobel_thresh(img,'y')

        mag_bin=mag_thresh(img, 3, (30, 100))
        
        dir_bin=dir_threshold(img,15, (0.7, 1.3))

        combined[((col_bin_r==1) &(col_bin_s==1)) | ((sobel_x_bin==1)&(mag_bin==1)) | ((dir_bin==1)&(sobel_x_bin==1))] = 1
    

    
    combined_binary=np.zeros(combined.shape[:2])

    combined_binary[(combined[:,:,0]==1) | (combined[:,:,1]==1)|(combined[:,:,2]==1)]=1


    warped,M,indicator_lines=warp(combined_binary)

    #uncomment to see if the trapezoidal is OK
    # line_image_blank = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    # out_img = np.asarray(np.dstack((warped, warped, warped)), np.float64)
    # draw_lines(line_image_blank, indicator_lines, color=[255, 0, 0], thickness=10)
    # line_image_blank=np.asarray(line_image_blank, np.float64)
    # print out_img.shape, line_image_blank.shape
    # lines_overlayed_image=cv2.addWeighted(out_img, 0.8, line_image_blank, 1., 0.)
    # return lines_overlayed_image
    

    Lane=fit_polynomial(warped, Lane)

    res,Lane=warp_back(warped,M,Lane)

    Lane=measure_curvature_pixels(res.shape,Lane)


    # VIDEO Printing

    font = cv2.FONT_HERSHEY_SIMPLEX

    org = (50, 50)

    fontScale = 1
   
    color = (0, 0, 200) 

    thickness = 2

    line_image_blank = np.zeros((img_distorted.shape[0], img_distorted.shape[1],3), dtype=np.uint8)

    left_mid_x,left_mid_y=int(Lane.left_line.pointx[-1]),620

    right_mid_x,right_mid_y=int(Lane.right_line.pointx[0]),620



    indicator_line_x=(left_mid_x+right_mid_x)//2

    indicator_line_y_bottom=(left_mid_y+right_mid_y)//2+30

    indicator_line_y_top=(left_mid_y+right_mid_y)//2-30

    off_set_lines=[[[ left_mid_x, left_mid_y,right_mid_x,right_mid_y],[indicator_line_x, indicator_line_y_top, indicator_line_x,indicator_line_y_bottom]]]  


    draw_lines(line_image_blank, off_set_lines, color=[5, 100, 150], thickness=3)

    return_image=shadow_mix(shadow_mix(img_distorted,res),line_image_blank,1,1)

    curv_x_text='Radius of Curvature: '+ str(round(Lane.right_line.radius_of_curvature,2))+ " Meters" if Lane.right_line.radius_of_curvature<3000 else "Strait Lines"

    cv2.putText(return_image, curv_x_text, org, font,  fontScale, color, thickness, cv2.LINE_AA)

    offset_text= "Offset: "+str(int(abs(Lane.center)*3.7/7))+ " Centimeters to the " +("right" if Lane.center<0 else "left")

    cv2.putText(return_image,offset_text, (indicator_line_x-100,indicator_line_y_top-25), font,  fontScale, (255,0,0), thickness, cv2.LINE_AA)

    return return_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def get_warp_params(img):


    src_corners=[[595,450],[690,450],[200,img.shape[0]],[1140,img.shape[0]]]#,

    #short, not working for all
    #src_corners=[[449,550],[857,550],[200,img.shape[0]],[1140,img.shape[0]]],

    # # another
    #src_corners= [[552,480],[740,480],[200,img.shape[0]],[1140,img.shape[0]]]

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

def warp_back(warped,m,Lane):

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([Lane.left_line.allx, Lane.left_line.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([Lane.right_line.allx, Lane.right_line.ally])))])
    pts = np.hstack((pts_left, pts_right))

    Lane.left_line.pointx=pts_left[0,:,0]

    Lane.right_line.pointx=pts_right[0,:,0]

    Lane.left_line.pointy=pts_left[0,:,1]

    Lane.right_line.pointy=pts_right[0,:,1]

    Lane.center=(pts_right[0][-1][0]+pts_left[0][-1][0])//2-warped.shape[1]//2

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    invertible,m_inv=cv2.invert(m)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (color_warp.shape[1], color_warp.shape[0])) 

    return newwarp,Lane


def find_lane_pixels(binary_warped, draw_windows=False):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)


    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50


    window_height = np.int(binary_warped.shape[0]//nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base


    left_lane_inds = []
    right_lane_inds = []


    for window in range(nwindows):

        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low = rightx_current - margin  
        win_xright_high = rightx_current + margin 
        

        if draw_windows:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),   (win_xleft_high,win_y_high),(0,255,0), 2) 

            cv2.rectangle(out_img,(win_xright_low,win_y_low),   (win_xright_high,win_y_high),(0,255,0), 2) 
        

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        
        if (len(good_left_inds)>minpix):
            leftx_current=int(np.mean(nonzerox[good_left_inds]))
        if (len(good_right_inds)>minpix):
            rightx_current=int(np.mean(nonzerox[good_right_inds]))    

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def search_around_poly(binary_warped,Lane):

    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_fit_gloabl_avg=np.average(Lane.left_line.recent_fits,axis=0)
    right_fit_gloabl_avg=np.average(Lane.right_line.recent_fits,axis=0)

    left_lane_inds = (np.absolute(np.polyval(left_fit_gloabl_avg,nonzeroy)-nonzerox)<margin).nonzero()[0]

    right_lane_inds = (np.absolute(np.polyval(right_fit_gloabl_avg,nonzeroy)-nonzerox)<margin).nonzero()[0]
    

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    return leftx, lefty, rightx, righty

def  fit_polynomial(binary_warped, Lane, parallel_thresh=0.0005, coef_change_thresh=0.1):



    if Lane.window_searched:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped, Lane) 
    else:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
        Lane.window_searched=True
    

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    if len(leftx)>0:# if no left line found
        Lane.left_line.current_fit = np.polyfit(lefty,leftx,2)# prefit because we want to check curvature before updating the global polyfit
        Lane.left_line.detected=True
    else:
        Lane.left_line.current_fit = np.array([0,0,binary_warped.shape[1]])

    if len(rightx)>0:
        Lane.right_line.current_fit = np.polyfit(righty,rightx,2)
        Lane.right_line.detected=True
    else:
        Lane.right_line.current_fit = np.array([0,0,binary_warped.shape[1]])

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )



    # decide to reset windowing or skip the current frame
    decision_factors_coef_sign= np.sign(Lane.left_line.current_fit[0])== np.sign(Lane.right_line.current_fit[0]) or ( not (Lane.right_line.detected and Lane.left_line.detected))

    decision_factors_coef_parallel=abs(Lane.right_line.current_fit[0]-Lane.left_line.current_fit[0])<parallel_thresh or ( not (Lane.right_line.detected and Lane.left_line.detected))

    decision_factors_right_line_reasonable_change=Lane.right_line.recent_fits and abs(Lane.right_line.recent_fits[-1][0]-Lane.right_line.current_fit[0])<coef_change_thresh

    decision_factors_left_line_reasonable_change=Lane.left_line.recent_fits and abs(Lane.left_line.recent_fits[-1][0]-Lane.left_line.current_fit[0])<coef_change_thresh

    
    left_fitx_bottom=Lane.left_line.current_fit[0]*ploty[-1]**2 + Lane.left_line.current_fit[1]*ploty[-1] + Lane.left_line.current_fit[2]

    right_fitx_bottom=Lane.right_line.current_fit[0]*ploty[-1]**2 + Lane.right_line.current_fit[1]*ploty[-1] + Lane.right_line.current_fit[2]

    decision_factors_bottm_point_left=left_fitx_bottom<binary_warped.shape[1]//2

    decision_factors_bottm_point_right=right_fitx_bottom>binary_warped.shape[1]//2

    left_fitx_top=Lane.left_line.current_fit[0]*ploty[0]**2 + Lane.left_line.current_fit[1]*ploty[0] + Lane.left_line.current_fit[2]

    right_fitx_top=Lane.right_line.current_fit[0]*ploty[0]**2 + Lane.right_line.current_fit[1]*ploty[0] + Lane.right_line.current_fit[2]

    decision_factors_top_points=right_fitx_top>left_fitx_top




    if Lane.first_iteration:
        Lane.first_iteration=False

        Lane.left_line.recent_fits.append(Lane.left_line.current_fit)

        Lane.right_line.recent_fits.append(Lane.right_line.current_fit)

    elif (decision_factors_coef_sign and 
    decision_factors_coef_parallel and 
    decision_factors_top_points and 
    decision_factors_right_line_reasonable_change and  
    decision_factors_bottm_point_right and 
    decision_factors_left_line_reasonable_change and 
    decision_factors_bottm_point_left):

        Lane.right_line.recent_fits.append(Lane.right_line.current_fit)
        Lane.left_line.recent_fits.append(Lane.left_line.current_fit)

    else:
        Lane.lost=True
        window_searched=True



    left_fit_gloabl_avg=np.average(Lane.left_line.recent_fits, axis=0)
    right_fit_gloabl_avg=np.average(Lane.right_line.recent_fits, axis=0)

    Lane.left_line.allx = left_fit_gloabl_avg[0]*ploty**2 + left_fit_gloabl_avg[1]*ploty + left_fit_gloabl_avg[2]
    Lane.left_line.ally=ploty

    Lane.right_line.allx = right_fit_gloabl_avg[0]*ploty**2 + right_fit_gloabl_avg[1]*ploty + right_fit_gloabl_avg[2]
    Lane.right_line.ally=ploty



    return Lane


def measure_curvature_pixels(image_shape,Lane,ym_per_pix = 30.0/720.0,xm_per_pix = 3.7/700.0): # meters per pixel in y dimension , # meters per pixel in x dimension
     
 
    
    ploty = np.linspace(0, image_shape[0]-1, num=image_shape[0])
    y_eval = np.max(ploty)


    left_fit_cr = np.polyfit(ploty*ym_per_pix, Lane.left_line.allx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, Lane.right_line.allx*xm_per_pix, 2)
    Al,Bl=Lane.left_line.current_fit[:2]
    Ar,Br=Lane.right_line.current_fit[:2]
    Lane.left_line.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    Lane.right_line.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
   
    return Lane

def shadow_mix(img1, img2, alpha=1, beta=0.3, gamma=0):
    
    # Combine the result with the original image
    result = cv2.addWeighted(img1, alpha, img2, beta, gamma)

    return result


def image_test(show=False):
    
    images = glob.glob('../test_images/*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:

        img=mpimg.imread(fname)

        L=Lane()

        L.image_class=2

        ret=pipeline(img,L)


        cv2.imwrite('../output_images/'+fname.split('/')[-1].split("8")[0], ret)
        
        if show:
            plt.imshow(ret)
        
            plt.show()
            
            cv2.waitKey(500)
            
            plt.close()


def classifier(video):
    # for now it's just a stupid fucntion
    return 0 if video!='harder_challenge_video' else 1

def video_test():
    videos=['project_video', 'challenge_video','harder_challenge_video']
    for i in range(len(videos)):#'project_video', 'challenge_video', 'harder_challenge_video'
        clip1 = VideoFileClip('../'+videos[i]+'.mp4')

   
        L=Lane()
        L.image_class=classifier(videos[i])
        white_clip = clip1.fl_image(lambda img:process_image(img,L))
        
        white_clip.write_videofile(videos[i]+'_processed.mp4', audio=False)


def process_image(img, Lane):

    return pipeline(img, Lane)








if __name__ == '__main__':
    

    # image_test()
    video_test()