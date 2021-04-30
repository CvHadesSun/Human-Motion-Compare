import json
import numpy as np
import os
import math
import cv2
import matplotlib.pyplot as plt
import copy
from datetime import timedelta
# path = './data/DATA_1509_CurtsyLungeLeft/data.json'
#
# data = {}
# with open(path,'r') as f:
#     data = json.load(f)
#
#     f.close()
#
# print(data.keys())
#
# data_frame = data['frames']
# print(len(data_frame[0]))
# print(data_frame[0].keys())
# #frame
# # dict_keys(['cameraData', 'modelBmpPath', 'modelData', 'result', 'time', 'timeStamp'])
#
# cam_data=data_frame[0]['cameraData']
# m_data = data_frame[0]['modelData']
#
# print(cam_data)
#
# print(m_data)


def getData(data):
    #data list
    np_data = np.zeros([len(data),5])
    for i in range(len(data)):
        norm_x =data[i]['x']
        norm_y = data[i]['y']
        angle = data[i]['angle']
        posx = data[i]['posX']
        posy = data[i]['posY']

        np_data[i,:] = norm_x,norm_y,angle,posx,posy

    return np_data
def getFrameData(dict):
    #get per frame info
    #return two class joints data
    cam_data = dict['cameraData']
    m_data = dict['modelData']

    #get cam data
    #return by cam_dict:{'image':'...','data':array[12,2+1+2]}

    cam_dict ={}
     #frame image name
    # tmp_data =
    cam_dict['data'] =[]
    cam_dict['image'] = []
    for item in cam_data:
        cam_dict['data'].append(getData(item['data']))
        cam_dict['image'].append(item['bmpPath'])
    # print(cam_dict['image'])

    #get model data
    m_dict = {}
    m_dict['image'] = dict['modelBmpPath']

    m_dict['data'] = getData(m_data)

    return cam_dict,m_dict

def draw2DJoint(joints,image):
    # image=copy.copy(image)
    print(image.shape)
    img = cv2.resize(image,(1080,1920))

    for i in range(joints.shape[0]):
        x,y=joints[i]
        if x>0 and y>0:
            cv2.circle(img, (int(1080-x), int(y)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    print(img.shape)
    cv2.imwrite('./test1.jpg', img)

    return image
def drawLimbs2D(joints,index,image,c=(0,255,0)):
    image=copy.copy(image)
    img = cv2.resize(image, (1080, 1920))
    # print(joints.shape)
    for ind in index:
        partA = ind[0]
        partB = ind[1]
        # try:
        # if joints[partA] and joints[partB]:
        y1=joints[partA][0]
        x1=joints[partA][1]
        y2=joints[partB][0]
        x2=joints[partB][1]
        # print(x1,y1,x2,y2)
        # if x1>0 and x2>0 and y1>0 and y2>0:
        # length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # # if length < 10000 and length > 5:
        # deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        # polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
        #                         (int(length / 2), 3),
        #                         int(deg),
        #                         0, 360, 1)
        # cv2.fillConvexPoly(image, polygon, color=c)
        cv2.line(img, (int(1080-y1), int(x1)), (int(1080-y2), int(x2)), c, 3)
        # except:
        #     continue
        cv2.imwrite('./test.jpg', img)
    return img

# def jointVis(img,joints,color=(0,255,0),index=[]):
#     if len(index)==0:
#         index = np.array([[1,2],[1,2],[3,5],[2,4],[4,6],
#                         [1,7],[2,8],[7,8],[7.9],[8,10],[9,11],[10,12]])
#         index = index-1
#     if img:
#         img_new = drawLimbs2D(joints,index,img,color)
#
#     else:

# def scoreJoints(cam_joints,m_joint):
    #

def findSim(cam_data,m_data):
    #cam_data: list
    # index=0
    min_value = 2.0
    # print(len(cam_data))
    for i in range(len(cam_data)):
        tmp = cam_data[i][:,:2]
        delta_dist = np.sum(np.sqrt(np.sum((m_data-tmp)**2,1)))/tmp.shape[0]
        if delta_dist<min_value:
            index=i
            min_value = delta_dist

    # print(index)
    return index

def concatImg(path,m_name,cam_names,index):
    m_img=cv2.imread(os.path.join(path,m_name))
    # print(m_img.shape)
    m_h,m_w,_ = m_img.shape
    h=int(1200)
    w=int(550)
    m_img= cv2.resize(m_img,(int(w-10),int(h-200)))
    num=len(cam_names)
    col=1
    if num<=5:
        col=1
    elif num>5 and num<=10:
        col=2
    else:
        col=3
    cam_h=int(240)
    cam_w=int(135)

    # #new img
    new_img = np.zeros([h,w+col*cam_w,3],dtype=np.uint8)
    #
    # #
    # # print(m_img)
    new_img[:h-200,:w-10,:] = m_img
    #
    # print(cam_names)
    for i in range(len(cam_names)):

        cam_img = cv2.imread(os.path.join(path,cam_names[i]))
        # print(cam_img)
        cam_img =cv2.resize(cam_img,(cam_w,cam_h))


        if i == index:
            cam_img[:10,:,:]=[0,255,0]
            cam_img[cam_h-10:,:,:] = [0,255,0]
            cam_img[:,:10,:] = [0,255,0]
            cam_img[:,cam_w-10:,:]=[0,255,0]

        #compute location
        bh=0
        bw=0
        if i <=4:
            bh=i*cam_h
            bw = 0
        elif i>4 and i<=9:
            bh=cam_h*(i-5)
            bw = 1*cam_w
        else:
            bh=cam_h*(i-10)
            bw=2*cam_w
        bh=int(bh)
        bw=int(bw)+w
        # print(bh,bw)
        new_img[bh:bh+cam_h,bw:bw+cam_w,:] = cam_img
    #
    new_name = 'comparation'+m_name
    cv2.imwrite(os.path.join(path,new_name),new_img)

def comparaJoint(m_data,cam_data,color):
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

def NormJoints(joints,root):

    #joints:all 12 joints
    #root: the origin for normlization
    length,_=joints.shape
    root_x,root_y=root
    norm_landmarks=np.zeros([length,2])
    # i=0
    # print(length)

    for ind in range(length):
        # print(joints[ind])
        x=joints[ind,0]
        y=joints[ind,1]
        # print(x,y)
        d=(x-root_x)**2+(y-root_y)**2
        x=(x-root_x)/math.sqrt(d)
        y=(y-root_y)/math.sqrt(d)
        norm_landmarks[ind,:]=[x,y]
        # i+=1
        # print(d)
    nn=NormJointsV2(joints,root)
    # print(nn)
    # print(norm_landmarks)
    # print(np.sum((joints- root)**2,axis=1))

    return nn


def flipJoint(data):
    posX= data[:,3]
    posY = data[:,4]
    #
    posX = 1080-posX
    data[:,3] = posX
    normjoints = NormJoints(data[:,3:],GetRoot(data[:,3:]))
    angle = AngleGen(normjoints).reshape(-1)

    data[:,0:2] = normjoints
    data[:,2] = angle

    return data


def main(json_path):
    data = {}
    with open(json_path, 'r') as f:
        data = json.load(f)

        f.close()
    all_dist = []
    all_angle = []
    # for frame in data['frames'][:3]:
    # for i in range(0,len(data['frames']),1):

    # for i in range(0,len(data['frames']),1):
    while True:
        i=7
        frame = data['frames'][i]
        cam,m = getFrameData(frame)
        m_data = m['data']
        m_data = flipJoint(m_data)
        m_data_angle = m_data[:, 2]
        m_data_joints = m_data[:, :2]
        index = findSim(cam['data'],m_data_joints)
        cam_data = cam['data'][index]
        cam_name = cam['image'][index]
        print(m['image'],index)
        # print(frame['modelBmpPath'])

        #concatenate imgs for vis
        # vis_img=concatImg('./data/DATA_1509_CurtsyLungeLeft',m['image'],cam['image'],index)
        # cv2.imsave(os.path.join())

        #joints vis and comparation
        _=visJoints(m_data,cam['data'],index)
        # parents_index = np.array([[1,2],[1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[8,10],[9,11],[10,12]],dtype=np.int8)
        # parents_index-=1
        #
        # img= cv2.imread(os.path.join('./data/DATA_1509_CurtsyLungeLeft/',m['image']))
        #
        # image=drawLimbs2D(m_data[:,3:],parents_index,img)

        # image =draw2DJoint(m_data[:,3:],img)
        # cv2.circle(img, (int(546/2), int(956/2)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
        # cv2.imwrite('./test.jpg', image)


        cam_data_joints = cam_data[:,:2]
        cam_data_angle = cam_data[:,2]




        #
        delta_dist = np.sqrt(np.sum((cam_data_joints-m_data_joints)**2,1)).reshape(-1,1)

        delta_angle = np.abs(cam_data_angle-m_data_angle).reshape(-1,1)
        all_dist.append(delta_dist)
        all_angle.append(delta_angle)
        break


    np_dist = np.concatenate(all_dist,axis=1)
    np_angle = np.concatenate(all_angle,axis=1)

    return np_dist,np_angle


path = './data/DATA_1509_CurtsyLungeLeft/data.json'

joints,angle = main(path)

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# color1 = '#0085c3'
# color2 = '#7ab800'
# color3 = '#dc5034'
# for i in range(0,joints.shape[1],2):
#     x = [i for i in range(joints.shape[0])]
#     y = joints[:,i]
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111)
#     ax.plot(x, y, marker='o', color=color1)
#     # ax.text(x[y.idxmax()] + timedelta(hours=-12),
#     #         y.max() + 1, y.max(),
#     #         color=color1, fontsize=15)
#
#     # 标注最小值
#     # ax.text(x[y.idxmin()] + timedelta(hours=-9),
#     #         y.min() - 2, y.min(),
#     #         color=color1, fontsize=15)
#     y2 = np.sum(y)/joints.shape[0]*(1+1/1)
#     # ax.plot(x, y2, ls='--', color=color2, label='7 天移动平均')
#
#     ax.hlines(y2, x[0], x[-1:],
#           linestyles='-.', colors=color3)
#     # ax.text(x[-1:] + timedelta(days=-7.5), y.mean() - 2,
#     #         '平均值: ' + str(round(y.mean(), 1)),
#     #         color=color3, fontsize=15)
#     ax.grid(ls=':', color='gray', alpha=0.6)
#     ax.legend(loc='upper left', fontsize=12)
#     plt.xticks(rotation=90, fontsize=12)
#     plt.yticks(fontsize=12)
#     # plt.show()
#     # break





# #json data format
# {
#     'frames':[
#         {
#             'camData':{'img':'xx.jpg'
#                        'data':[{x: 1, y: 1,angle:1,posX: 1,posY: 1}, ...]} #most similar frame or time correspond frame in cam data
#
#             'modelData':{'img':'xx.jpg',
#                        'data':[{x: 1,y: 1,angle:1,posX: 1,posY: 1}, ...]} #corresponding model coach frame
#         }
#     ]
# }
#
#
# #数据处理一致（水平反转操作，其他）







