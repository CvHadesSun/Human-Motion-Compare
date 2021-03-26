from utils import *

# process coach motion and save data norm joints and angles into txt file

StandMotionPorcess('data/action-1.mp4',
                   'data/action-1', mode='torso')

# process the in time frames flow


# define weight
weight = np.ones([12, 3])

weight[8:, -1] = 0
# print(weight)
d, a = ProcessVideo('data/3-action-test.mp4',
                    'data/action-1/norm-action.txt',
                    12, 30, weight, mode='torso')


# for vis
x = [i for i in range(len(d))]
plt.plot(x, d)
plt.plot(x, a)
plt.show()
