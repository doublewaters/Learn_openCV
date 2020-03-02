
# display a video from a file

#  CV_CAP_PROP_POS_MSEC Current position of the video ﬁle in milliseconds. 
#  CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next. 
#  CV_CAP_PROP_POS_AVI_RATIO Relative position of the video ﬁle: 0 - start of the ﬁlm, 1 - end of the ﬁlm. 
#  CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream. 
#  CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream. 
#  CV_CAP_PROP_FPS Frame rate. • CV_CAP_PROP_FOURCC 4-character code of codec. 
#  CV_CAP_PROP_FRAME_COUNT Number of frames in the video ﬁle. 
#  CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() . 
#  CV_CAP_PROP_MODE Backend-speciﬁc value indicating the current capture mode. 
#  CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras). 
#  CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras). 
#  CV_CAP_PROP_SATURATION Saturation of the image (only for cameras). 
#  CV_CAP_PROP_HUE Hue of the image (only for cameras). 
#  CV_CAP_PROP_GAIN Gain of the image (only for cameras). 
#  CV_CAP_PROP_EXPOSURE Exposure (only for cameras). 
#  CV_CAP_PROP_CONVERT_RGB Boolean ﬂags indicating whether images should be converted to RGB. 
#  CV_CAP_PROP_WHITE_BALANCE Currently unsupported 
#  CV_CAP_PROP_RECTIFICATIONRectiﬁcationﬂagforstereo cameras (note: only supported by DC1394 v 2.x backend currently)


import numpy as np
import cv2 

cap = cv2.VideoCapture(0) 

# Define the codec and create VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,0)

        #write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

# Release everything if job is finished 
cap.release()
out.release()
cv2.destroyAllWindows()