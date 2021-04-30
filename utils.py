import cv2
import mediapipe as mp
import  os
import numpy as np
import math
import matplotlib.pyplot as plt

import time


def NormJoints(joints,root):

    #joints:all 12 joints
    #root: the origin for normlization
    length,_=joints.shape
    root_x,root_y=root
    norm_landmarks=np.zeros([length,2])
    i=0
    for ind in range(length-1):
        x=joints[ind,0]
        y=joints[ind,1]
        d=math.sqrt((root_x-x)**2+(root_y-y)**2)
        x=(x-root_x)/d
        y=(y-root_y)/d
        norm_landmarks[i,:]=[x,y]
        i+=1

    return norm_landmarks


def AngleGen(norm_joints):
    #there are 8 angles between 12 joints
    #angle 1: (3,1,7)
    angle1=Vector2Angle(norm_joints[2],norm_joints[0],norm_joints[6])
    #angle 2: (4,2,8)
    angle2=Vector2Angle(norm_joints[3],norm_joints[1],norm_joints[7])
    #angle 3: (5,3,1)
    angle3=Vector2Angle(norm_joints[4],norm_joints[2],norm_joints[0])
    #angle 4: (2,4,6)
    angle4=Vector2Angle(norm_joints[1],norm_joints[3],norm_joints[5])
    #angle 5: (1,7,9)
    angle5=Vector2Angle(norm_joints[0],norm_joints[6],norm_joints[8])       
    #angle 6: (2,8,10)
    angle6=Vector2Angle(norm_joints[1],norm_joints[7],norm_joints[9])
    #angle 7: (7,9,11)
    angle7=Vector2Angle(norm_joints[6],norm_joints[8],norm_joints[10])
    #angle 8: (8,10,12)
    angle8=Vector2Angle(norm_joints[7],norm_joints[9],norm_joints[11])

    return np.array([angle1,angle2,angle3,
                    angle4,angle5,angle6,
                    angle7,angle8,0,0,0,0]).reshape([-1,1])


def Vector2Angle(v1,v0,v2):
    #
    v1_x=v1[0]-v0[0]
    v1_y=v1[1]-v0[1]
    v2_x=v2[0]-v0[0]
    v2_y=v2[1]-v0[1]
    #
    cos_thelta=(v1_x*v2_x+v1_y*v2_y)/(math.sqrt(v1_x**2+v1_y**2)*math.sqrt(v2_x**2+v2_y**2))
    return math.acos(cos_thelta)

def NormJointsV2(joints,root):
    #more efficient
    num_joints, _ = joints.shape
    return  (joints-root)/np.sqrt(np.sum((joints- root)**2,axis=0).reshape(-1))

def GetRoot(joints,mode='torso'):
    #mode 1:left_hip
    #mode 2:right_hip
    #mode 3:torso (the middle of left_shoulder,right_shoulder,left_hip and right_hip)
    if mode not in ['left_hip','right_hip','torso']:
        print("error: no this mode")
        return joints[0]
    if mode=='left_hip':
        return joints[7]
    elif mode=='right_hip':
        return joints[6]
    else:
        x=joints[0,0]+joints[1,0]+joints[6,0]+joints[7,0]
        y=joints[0,1]+joints[1,1]+joints[6,1]+joints[7,1]
    return np.array([x/4,y/4])



def StandMotionPorcess(video_path,save_txt_path,debug_output=False,save_imgs_path='images',mode='torso'):
    #process coach motion and save norm_joints and angles
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    joints=np.zeros([12,2])
    frame=0
    all_frames=[]
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          break
          # continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        # mp_drawing.draw_landmarks(
        #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        #get  coordinates

        # joints_names=[mp_holistic.PoseLandmark.LEFT_SHOULDER,
        #              mp_holistic.PoseLandmark.RIGHT_SHOULDER,
        #              mp_holistic.PoseLandmark.LEFT_ELBOW,
        #              mp_holistic.PoseLandmark.RIGHT_ELBOW,
        #              mp_holistic.PoseLandmark.LEFT_WRIST,
        #              mp_holistic.PoseLandmark.RIGHT_WRIST,
        #              mp_holistic.PoseLandmark.LEFT_HIP,
        #              mp_holistic.PoseLandmark.RIGHT_HIP,
        #              mp_holistic.PoseLandmark.LEFT_KNEE,
        #              mp_holistic.PoseLandmark.RIGHT_KNEE,
        #              mp_holistic.PoseLandmark.LEFT_ANKLE,
        #              mp_holistic.PoseLandmark.RIGHT_ANKLE
        #              ]
        joints[0, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width
        joints[0, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height

        joints[1, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width
        joints[1, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height

        joints[2, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width
        joints[2, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_height

        joints[3, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width
        joints[3, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image_height

        joints[4, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image_width
        joints[4, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image_height

        joints[5, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image_width
        joints[5, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_height

        joints[6, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width
        joints[6, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height

        joints[7, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width
        joints[7, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height

        joints[8, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width
        joints[8, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height

        joints[9, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x * image_width
        joints[9, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y * image_height

        joints[10, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].x * image_width
        joints[10, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].y * image_height

        joints[11, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].x * image_width
        joints[11, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].y * image_height


        #extend timestamp
        # t1=time.time()

        #
        root = GetRoot(joints,mode)
        norm_joints=NormJoints(joints,root)
        angles=AngleGen(norm_joints)
        angle_joints=np.concatenate([norm_joints,angles],axis=1)
        all_frames.append(angle_joints)
        if debug_output:
            cv2.imwrite(os.path.join(save_imgs_path,str(frame)+'.jpg'),image)
            # continue
        frame+=1

        # cv2.imshow('MediaPipe Holistic', image)
        # if cv2.waitKey(5) & 0xFF == 27:
        #   break
      np_frames=np.concatenate(all_frames,axis=0)
      np.savetxt(os.path.join(save_txt_path,'norm-action.txt'),np_frames)

    cap.release()
    cv2.destroyAllWindows()



def ProcessVideo(video_path,action_path,num_joints,frame_ratio,weight,save_path,debug_ouput=False,mode='torso'):
    if debug_ouput:
        mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    joints=np.zeros([num_joints,2])
    action=np.loadtxt(action_path)
    num_frames=action.shape[0]/num_joints
    time_length=num_frames/frame_ratio
    frame=1
    time_stamps=[]
    processed_frames=[]
    processed_imgs=[]
    d=[]
    angle=[]
    t0=time.time()

    #cvhadessun add 
    t_sample = time.time()
    time_interval = 1/5
    #
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          break
        t_current = time.time()  #cvhadessun add 
        if t_current-t_sample<time_interval:
            continue
        t_sample = t_current
        #

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        t1=time.time()
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        if debug_ouput:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        #get  coordinates
        joints[0, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width
        joints[0, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height

        joints[1, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width
        joints[1, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height

        joints[2, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width
        joints[2, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_height

        joints[3, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width
        joints[3, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image_height

        joints[4, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image_width
        joints[4, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image_height

        joints[5, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width
        joints[5, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_height

        joints[6, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width
        joints[6, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height

        joints[7, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width
        joints[7, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height

        joints[8, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width
        joints[8, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height

        joints[9, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x * image_width
        joints[9, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y * image_height

        joints[10, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].x * image_width
        joints[10, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].y * image_height

        joints[11, 0] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].x * image_width
        joints[11, 1] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].y * image_height

        root = GetRoot(joints,mode)
        norm_joints=NormJoints(joints,root)
        angles=AngleGen(norm_joints)
        angle_joints=np.concatenate([norm_joints,angles],axis=1)

        processed_frames.append(angle_joints)
        time_stamps.append(t1-t0)
        if debug_ouput:
            processed_imgs.append(image)
        
        #time
        
        # print(t1-t0)
        if t1-t0>time_length:
            np_frames = np.concatenate(processed_frames, axis=0)
            _d,_angle=ProcessOnePeriod(np_frames,action,weight,time_stamps,time_length,frame_ratio,3)
            d.append(_d)
            angle.append(_angle)
            print(_angle)
            #
            frame=0
            processed_frames=[]
            time_stamps=[]
            t0=time.time()



        frame += 1



    cap.release()
    cv2.destroyAllWindows()
    return d,angle




def AssignWeight(joints_angle1,joints_angle2,weight):
    #joints_angle:shape[12,3] numpy
    # weight:shape[12,2] numpy

    weight_joints=weight[:,0]
    weight_angles=weight[:,1]

    _joints1 = joints_angle1[:,:-1]
    _angles1 = joints_angle1[:,-1]

    _joints2 = joints_angle2[:,:-1]
    _angles2 = joints_angle2[:,-1]
    
    #joints distance
    d=np.sum((_joints1-_joints2)**2,axis=1)
    d=np.sqrt(d)
    d_weight=d*weight_joints

    delta_angle=abs(_angles1-_angles2)*weight_angles

    # cos_d=math.cos(np.sum(d_weight)[0])
    d=np.sum(d_weight)/joints_angle1.shape[0]
    mean_angle=np.sum(delta_angle)/8
    # print(mean_angle)


    return d,mean_angle


def TimeStamp2Index(time_stamp,time_length,ratio_frame=30,drop=3):

    #tranform the time stamp to index for comparing the same time frame

    #time_stamp :[timestamp1,timestamp2,....] :unit(s) [0,time_length]
    #time_lenght: the length of compare time :unit(s)
    #ratio_frame: the coach video frame sample ratio

    #can drop some timestamp in the begin of one time  comparing or drop in the final return 
    _indexs=[]
    interval=1/ratio_frame

    # print(time_stamp)
    for t in time_stamp:
        _indexs.append(t//interval)
    #can drop some begin index for alignment
    indexs=_indexs[drop:-drop+1]
    return indexs   
        
def ProcessOnePeriod(precessed_frames,stand_frames,weight,timestamps,time_lenght,ratio_frame,drop):
    #processed_frames:need to be scored
    #stand_frames:coach 's motion
    #drop: to smooth the begin and end of motion for tow time sequentially motion
    num_joints=12
    #reshape
    stand_frames=stand_frames.reshape([-1,num_joints,3])
    precessed_frames=precessed_frames.reshape([-1,num_joints,3])
    precessed_frames=precessed_frames[drop:-drop+1,:,:]
    index=TimeStamp2Index(timestamps,time_lenght,ratio_frame,drop)
    d=0
    angle=0
    for i,ind in enumerate(index):
        _d,_angle=AssignWeight(stand_frames[int(ind)],precessed_frames[i],weight)
        d+=_d
        angle+=_angle

    return d/len(index),angle/len(index)

def comparaJoint(m_data,cam_data,color='#DC143C'):
    colors1 = '#00CED1'
    # colors2 = '#DC143C'
    # colors3 = '#000000'

    area = np.pi * 4 ** 2
    #
    # plt.gca().invert_yaxis()
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    m_root = GetRoot(m_data[:,3:])
    cam_root = GetRoot(cam_data[:,3:])

    cam_data[:,0:2] = NormJointsV2(cam_data[:,3:],cam_root)
    # m_root=[0,0]
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.scatter(int(cam_root[0]), -int(cam_root[1]), s=area, c='#000000', alpha=0.4, label='m')
    plt.scatter(m_data[:,0], -m_data[:,1], s=area, c=colors1, alpha=0.4, label='m')
    plt.scatter(cam_data[:, 0], -cam_data[:, 1], s=area, c=color, alpha=0.4, label='cam')
    plt.show()
    return True


def visJoints(m_data,cam_data,index):
    # color = ''
    colors2 = '#DC143C'
    colors3 = '#000000'
    for i in range(len(cam_data)):
        if i == index:
            color = '#DC143C'
        else:
            color = '#000000'

        #c
        _=comparaJoint(m_data,cam_data[i],color)
    return True


def jointVote(cam_norm_data,m_norm_data,params_dist):
    '''to get the vote result in joints ;
    @input:
        cam_norm_data:the norm coordinates in camera frame [12,2]([x,y])
        m_norm_data: the norm coordinates in coach frame [12,2]([x,y])
        params_dist: the scoring base params of distance between the two corresponding frame joints. [k,alpha,beta]
    @output:
        results: the vote result of Euclidean distance between the two corresponding frame joints with params_dist.[12,]
    '''
    k,alpha,beta = params_dist
    #compute the Euclidean distance
    delta_dist = np.sqrt(np.sum((cam_norm_data-m_norm_data)**2,axis=1)) #((x1-x2)^2+(y1-y2)^2)^1/2
    #first: use delta_dist minus threshold alpha;
    results = np.zeros([delta_dist.shape[0]])
    # the index of joints of Euclidean distance in [0,alpha]
    ind = np.where((delta_dist-alpha)<=0)
    # results[ind] = 1.0
    delta_dist = delta_dist-alpha
    delta_dist[ind] = 0.0
    results = np.exp(-k*(delta_dist)) - beta  #y = exp(-kx) -b
    results[ind] = 1.0

    return results

def angleVote(cam_angle,m_angle,params_angle):
    '''to get the vote result in angle;
    @input:
        cam_angle:the norm vector angle in camera frame [12,]  ,the angle is in [0,pi]
        m_angle:the norm vector angle in coach frame [12,]  ,the angle is in [0,pi]
        params_angle: the scoring base params of distance between the two corresponding frame angle. [k,alpha,beta]
    @output:
        results: the vote result of delta angle with params_angle,[8,]
        '''

    k,alpha,beta = params_angle

    #
    delta_angle = np.abs(cam_angle-m_angle)[:8]

    #
    ind = np.where((delta_angle-alpha)<=0)
    delta_angle =delta_angle-alpha
    delta_angle [ind] = 0.0

    results = np.exp(-k*delta_angle) - beta

    results[ind] = 1.0

    return results


def assignWeight(joints_scale,angle_scale,weight):
    '''to assign the weight on the joints and angle parameters and get the final score between
    two motion;
    @input:
        joints_scale:the vote results of joints ; [12,]
        angle_scale: the vote results of angle  ;[8,]
        weight: the scale of (joints_propotion/angle_propotion)
    @output:
        final_score: a score to determinate the similarity between two motion  (total score is 100.0)
    '''
    joints_score = 100.0 * (1-1/(weight+1))
    angle_score = 100.0 * (1/(weight+1))

    joints_score = np.sum(joints_score/12 * joints_scale)
    angle_score = np.sum(angle_score / 8 * angle_scale)

    return joints_scale+angle_score

