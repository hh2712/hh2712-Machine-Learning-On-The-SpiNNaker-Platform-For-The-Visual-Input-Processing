__author__ = "hanyi"
import scipy.io
import math
argw = []

prefix_l_t_r = 'l_to_r'
prefix_r_t_l = 'r_to_l'
prefix_t_to_b = 't_to_b'
prefix_b_to_t = 'b_to_t'
post_fix = '.mat'

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0])/1000 #from us to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts

for i in range(1,10):
    argw.append(get_data(prefix_l_t_r+str(i)+post_fix))

for i in range(1,10):
    argw.append(get_data(prefix_r_t_l+str(i)+post_fix))
    
for i in range(1,10):
    argw.append(get_data(prefix_t_to_b+str(i)+post_fix))
    
for i in range(1,10):
    argw.append(get_data(prefix_b_to_t+str(i)+post_fix))

ONOFF  = 1
count1 = 0
count2 = 0
count3 = 0
count4 = 0   
for i in range(36):
    count = 0
    x1 = []
    y1 = []
    p1 = []
    ts1 = []
    for j in range(len(argw[i][2])):
        if (argw[i][2][j] == ONOFF):
            count = count + 1
            x1.append(argw[i][0][j])
            y1.append(argw[i][1][j])
            p1.append(argw[i][2][j])
            ts1.append(argw[i][3][j])
    x2 = []
    y2 = []
    p2 = []
    ts2 = []
    k = 0
    step = count/89.0    
    j = 0.0
    temp = 0
    while(j<count):
        if(temp != int(math.floor(j))):
            x2.append(x1[int(math.floor(j))])
            y2.append(y1[int(math.floor(j))])
            p2.append(p1[int(math.floor(j))])
            ts2.append(ts1[int(math.floor(j))])
            temp = int(math.floor(j))
        j = j+step
        k = k+1
        
    if(i<9):
        scipy.io.savemat(prefix_l_t_r+str(i+1)+post_fix,{'ts':ts2,
                                       'X':x2,
                                       'Y':y2,
                                       't':p2
                                       })
    if((i<18) and (i>=9)):
        scipy.io.savemat(prefix_r_t_l+str(i-8)+post_fix,{'ts':ts2,
                               'X':x2,
                               'Y':y2,
                               't':p2
                               })
    if((i<27) and (i>=18)):
        scipy.io.savemat(prefix_t_to_b+str(i-17)+post_fix,{'ts':ts2,
                               'X':x2,
                               'Y':y2,
                               't':p2
                               })
    if((i<36) and (i>=27)):
        scipy.io.savemat(prefix_b_to_t+str(i-26)+post_fix,{'ts':ts2,
                               'X':x2,
                               'Y':y2,
                               't':p2
                               })
    