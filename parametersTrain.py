
#@author:cvhadessun-2021.4.26
#process camera data and model data to train the most suitable parameters for motion scoring

import json
import os
import numpy as np
#
from utils import *

# global config

debug = True
norm_again = True
mode = 'torso' # ['left_hip','right_hip','torso']
#file path
root = './Similarity-0428'

params_dist = [1,0.1,0.2]
params_angle = [1,0.2,0.2]
weight = 4/3

#
way = 1 # 0:jointVote 1:jointVoteWithDist

def trainParam(cam_data,m_data,params_dist,params_angle,weight):
    ''' to train parameters;
    @input:
        ccam_data: a frame of camera data [12,5] ([norm_x,norm_y,angle,posX,posY])
        m_data:corresponding camera frame(coach data)[12,5] ([norm_x,norm_y,angle,posX,posY])
        params_dist: the params of joints
        params_angle: the params of angles
        weight: for weight the score between joints and angles (example:4/3,...)
    @output:
    '''
    #
    if norm_again:
        #norm cam and m data
        cam_data[:,0:2] = NormJointsV2(cam_data[:,3:],GetRoot(cam_data[:,3:]))
        cam_data[:,2] = AngleGen(cam_data[:,0:2]).reshape(-1)

        cam_data[:, 0:2] = NormJointsV2(cam_data[:, 3:], GetRoot(cam_data[:, 3:]))
        cam_data[:, 2] = AngleGen(cam_data[:,0: 2]).reshape(-1)

        #
    if debug:
        comparaJoint(cam_data,m_data)
    if way:
        S_joints = jointVoteWithDist(cam_data[:, 0:2], m_data[:, 0:2], params_dist)
    else:
        S_joints = jointVote(cam_data[:,0:2],m_data[:,0:2],params_dist)
    S_angle = angleVote(cam_data[:,2],m_data[:,2],params_angle)
    score = assignWeight(S_joints,S_angle,weight)

    return score
def list2numpy(l):
    '''[{x,y,angle,posX,posY},...] to array[12,5]'''

    np_list = np.zeros([len(l),5])
    for i in range(len(l)):
        np_list[i,:] = l[i]['x'],l[i]['y'],l[i]['angle'],l[i]['posX'],l[i]['posY']

    return np_list

def readDate(file_dir,params_dist,params_angle,weight):
    json_file =os.path.join(root,file_dir,'data.json')
    data={}
    with open(json_file,'r' ) as  f:
        data = json.load(f)
        f.close()
    # print(data.keys())
    # print(data['frames'][0]['cameraData'])
    # print(data['frames'][0].keys())

    frames =  data['frames']
    scores = []

    for frame in frames:
        #coach data
        m_image_name = frame['modelBmpPath']
        m_data = list2numpy(frame['modelData'])

        #camera data
        cam_image_name = frame["cameraData"][0]['bmpPath']
        cam_data = list2numpy(frame['cameraData'][0]['data'])

        #compute the score

        scores.append(trainParam(cam_data,m_data,params_dist,params_angle,weight))

    return scores








readDate('DATA_1706_HighPlankKneetoElbow',params_dist,params_angle,weight)






