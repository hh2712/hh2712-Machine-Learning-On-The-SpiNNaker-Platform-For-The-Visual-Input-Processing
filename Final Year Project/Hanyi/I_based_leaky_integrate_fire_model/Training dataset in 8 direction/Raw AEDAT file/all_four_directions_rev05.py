import paer
import numpy as np

base_dir = 'C:\Python27\python-aer-0.1.3'
file = '315d_10.aedat'
matfile = '315d_4.mat'
min = 175861
max = 509147
# Each ball movement should be .5s long
animation_time = 0.2

# 3,280 events per second for 16*16 is reasonable for ball movement (might be even too high!)
num_events_p_s = 5000

# Helper function to read a file. Given (min,max) which are data ranges for extraction, this will return a cropped and
#  suitably sparse output.
def get_data(file, min, max, animation_time=animation_time, num_events=num_events_p_s*animation_time, offset=0):
    aefile = paer.aefile(file, max_events=max+1)
    aedata = paer.aedata(aefile)
    #print 'Points: %i, Time: %0.2f. End Time: %0.2f. Start Time: %0.2f.Sparsity: %i' % (len(aefile.data), (aefile.timestamp[-1]-aefile.timestamp[0])/1000000, aefile.timestamp[-1]/1000000, aefile.timestamp[0]/1000000,
    #                                              np.floor(len(aefile.data)/num_events))

    sparse = aedata[min:max].make_sparse(np.floor((max-min)/num_events))
    for i in range(1,len(sparse.ts)):
        #print sparse.y[-i]
        if(sparse.ts[-i]!=0):
            last_index = -i
            break
    print "last index: %d, last time stamp %d"%(last_index,sparse.ts[last_index])       
    actual_time = (sparse.ts[last_index]-sparse.ts[0])/1000000
    scale = actual_time/animation_time
    sparse.ts = (offset * 1000000) + np.round((sparse.ts-sparse.ts[0])/scale)
    # print sparse_ts[0], sparse_ts[-1], sparse_ts[-1]-sparse_ts[0], (sparse_ts[-1]-sparse_ts[0])/1000000
    print len(sparse[0:len(sparse.ts)+last_index+1])
    return sparse[0:len(sparse.ts)+last_index+1]
    #return aedata

# Loop through all files - indexes are extrapolated.
x1 = get_data(file, min, max, offset=0*animation_time)

# Need to pre-load a file, to get the correct headers when writing!
#lib = paer.aefile(file, max_events=1)

#lib.save(final, 'test.aedat')
print "file:"+file
print "matfile:" + matfile
d1 = x1.downsample((16,16))
#lib.save(d1, 'test_16.aedat')
d1.save_to_mat(matfile)
