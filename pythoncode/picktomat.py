
import sys
import scipy.io

#quick function to turn my pickle files into mat files
import pickle

fname = sys.argv[1]
fname_mat = fname[0:-1]+'mat'
alldata = pickle.load( open( fname, "rb" ) )
index = alldata[0]
x_estvec = alldata[1]
x_targvec = alldata[2]
PI_estvec = alldata[3]
contall = alldata[4]
loss = alldata[5]
loss_nocont = alldata[6]

#add additional terms here

mdict = {}
for k in range(len(alldata[0])):
    mdict.update({alldata[0][k]:alldata[k+1]})

scipy.io.savemat(fname_mat, mdict)
