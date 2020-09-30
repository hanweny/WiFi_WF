import train_nn
import cv2
import sys
import os
import operator
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def RunNN(divisor,xr,yr,model, geo_num = None):
    x_mesh = (xr[1] - xr[0]) // divisor
    y_mesh = (yr[1] - yr[0]) // divisor
    if not geo_num:
        geo_num = 1
    spec_img_path = 'maps/out{}.png'.format(geo_num) if type(train_nn.GEO) is not str else os.path.join(train_nn.GEO, 'out{}.png'.format(geo_num))
    spec_img_path = train_nn.GEO if train_nn.SINGLE_GEO and os.path.isfile(train_nn.GEO) else os.path.join(train_nn.GEO, 'out1.png') if train_nn.SINGLE_GEO else spec_img_path
    img_scale = cv2.cvtColor(cv2.imread(spec_img_path), cv2.COLOR_BGR2RGB) 

    X_LIST = []
    ANT_LOC_LIST = []
    for x in np.arange(xr[0], xr[1] + 1, x_mesh):
        for y in np.arange(yr[0], yr[1] + 1, y_mesh):
            ANT_LOC = (x, y, 5); ANT_LOC_LIST.append(ANT_LOC)
            X_LIST.append(train_nn.construct_X_helper(ANT_LOC, img_scale))
    Y_DIM, X_DIM, Z_DIM = X_LIST[0].shape
    X_LIST = np.array(X_LIST).reshape(-1, Y_DIM, X_DIM, Z_DIM)
    Y = model.predict(X_LIST)
    return ANT_LOC_LIST, Y

def get_stats(ANT_LOC_LIST, Y):
    Y_temp = Y.copy()
    Y_temp = Y_temp.clip(min = 0)
    maxPw = Y_temp.max(axis = (1, 2)) * 5000
    threshold = -85
    score_func = lambda ai, mpi: 10 * np.log10(ai / mpi) - 30
    Y_temp = np.array([score_func(ai, mpi) for ai, mpi in zip(Y_temp, maxPw)])
    qualified_count = (Y_temp >= threshold).sum(axis = (1, 2))
    roomCoverage = qualified_count / (Y.shape[1] * Y.shape[2]) * 100
    return roomCoverage

def plot_signal_strength(df, antenna_xy):
    plt.clf()
    mpl.use("pdf")
    plt.figure(figsize=(10, 5))
    sc = plt.matshow(df, vmin=None, vmax=None, fignum = 1, cmap="coolwarm")
    plt.xticks(range(0, 25 , 2), np.arange(-7, 8, 2))
    plt.yticks(range(0, 41, 4), np.arange(10, -11, -2))
    plt.colorbar(sc)
    plt.savefig("PredMap"+str(antenna_xy)+".png")

def main():
    args = train_nn.parse_arguments()
    train_nn.set_global_variables(train_nn.master)
    print('\nX dimension:  {},    X range:  {}'.format(train_nn.X_DIM, train_nn.XRANGE))
    print("Y dimension:  {},    Y range:  {}\n".format(train_nn.Y_DIM, train_nn.YRANGE))

    ANT_LOC_LIST, Y = RunNN(4, train_nn.XRANGE, train_nn.YRANGE, train_nn.model)
    roomCoverage = get_stats(ANT_LOC_LIST, Y)
    best_loc_idx = np.argmax(roomCoverage)
    print("Room Coverage Score:  {}\n".format(roomCoverage[best_loc_idx]))
    plot_signal_strength(Y[best_loc_idx, :, :], ANT_LOC_LIST[best_loc_idx])
    print("Success\n")

if __name__ == '__main__':
    main()
