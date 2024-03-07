#%% Import Libraries.
import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt
import cv2



#%%
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# # VIDEO FEED
# # Below creates an object of VideoCapture
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     image = cv2.flip(frame, 1)
#     cv2.imshow('Mediapipe Feed', image)
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cv2.waitKey(10)       
# cap.release()
# cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)
# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# #     cv2.waitKey(5)
#     while cap.isOpened():
# #     for i in range(1):
#         ret, frame = cap.read()
#         frame = cv2.flip(frame,1)
        
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = pose.process(image)
        
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                  )               
        
#         cv2.imshow('Mediapipe Feed', image)
#         cv2.waitKey(5)
#         if cv2.waitKey(10) & 0xFF == ord('q'):

#     cap.release()
#     cv2.destroyAllWindows()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame = '/Users/saturn/Downloads/pose_image.jpg'
    image = cv2.imread(frame)
    image_ = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_.flags.writeable = False

    # Make detection
    results = pose.process(image_)

    # Recolor back to BGR
    image_.flags.writeable = True
    image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)


# cv2.waitKey(0)
cv2.destroyAllWindows()