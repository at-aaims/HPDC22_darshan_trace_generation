#!/user/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dtype", help="dtype", default="huge")
parser.add_argument("modelname", help="modelname", default="flow")
args = parser.parse_args()
dtype = args.dtype
modelname = args.modelname

if dtype == "huge":
    nlen, nclass, DIM = 26, 10, 75 + 7
if dtype == "large":
    nlen, nclass, DIM = 220, 10, 60 + 7
if dtype == "medium":
    # nlen, nclass = 945, 12
    nlen, nclass, DIM = 417, 7, 25 + 7

print(">>> Parameters")
for pname in [
    "dtype",
]:
    print("%s: %r" % (pname, eval(pname)))
print("")

lx = list()
ly = list()
lb = list()

for APP in range(nclass):
    try:
        x = np.load(
            "_train_x_%s_%d_DIM%d_app%d_%s.npy" % (dtype, nlen, DIM, APP, modelname)
        )
        y = np.load(
            "_train_y_%s_%d_DIM%d_app%d_%s.npy" % (dtype, nlen, DIM, APP, modelname)
        )

        lx.append(x)
        ly.append(y)
        lb.append([APP,] * len(x))

        print(APP, x.shape, x.sum())
    except:
        print("Error in reading:", dtype, APP)
        pass

xx = np.concatenate(lx)
yy = np.concatenate(ly)
ll = np.concatenate(lb)
print(xx.shape, yy.shape, ll.shape)

np.save("flow_train_x_%s_%d_DIM%d_%s.npy" % (dtype, nlen, DIM, modelname), xx)
np.save("flow_train_y_%s_%d_DIM%d_%s.npy" % (dtype, nlen, DIM, modelname), yy)
np.save("flow_train_lb_%s_%d_DIM%d_%s.npy" % (dtype, nlen, DIM, modelname), ll)
