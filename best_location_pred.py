import train_nn
import cv2, sys, os, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.models import load_model

def RunNN(divisor,xr,yr, img_scale, model):
    x_mesh = (xr[1] - xr[0]) // divisor
    y_mesh = (yr[1] - yr[0]) // divisor
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
    xr, yr = train_nn.XRANGE, train_nn.YRANGE
    mesh_x, mesh_y = train_nn.MESH_X, train_nn.MESH_Y
    plt.xticks(range(0, df.shape[1], 4), np.arange(xr[0], xr[1] + mesh_x * 4, 
               mesh_x * 4), rotation = 60, ha = 'left')
    plt.yticks(range(0, df.shape[0], 4), np.arange(yr[0], yr[1] + mesh_y * 4, 
               mesh_y * 4))
    plt.colorbar(sc)
    plt.savefig("PredMap"+str(antenna_xy)+".png")

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Predict the best locations for WiFi router installment")
    parser.add_argument("map_path", type = str,
                        help = "Preprocessed map file path")
    parser.add_argument("model_path", type = str,
                        help = "Trained model path")
    args = parser.parse_args()
    try:
        img_scale = cv2.cvtColor(cv2.imread(args.map_path), cv2.COLOR_BGR2RGB) 
    except:
        raise Exception("Preprocessed map file  does not exist at:  {}".format(args.map_path))
    try:
        model = load_model(args.model_path, custom_objects = {"coeff_determination":train_nn.coeff_determination})        
    except:
        raise Exception("Trained Model does not exist at:  {}".format(args.model_path))
    try:
        assert(img_scale.shape[0] == model.layers[0].input_shape[1] and \
               img_scale.shape[1] == model.layers[0].input_shape[2])
    except:
        raise Exception("Map dimension does not match with the model input shape. Examine the map and model pair")

    return img_scale, model

def set_train_nn_global_variables(img_scale):
    train_nn.VAL_MAX = 10**2
    train_nn.PIXEL_MARGIN_ERROR_THRESHOLD = 100
    train_nn.XRANGE = [0, img_scale.shape[1]-1]
    train_nn.YRANGE = [0, img_scale.shape[0]-1]
    train_nn.MESH_X, train_nn.MESH_Y = 1, 1
    train_nn.MESH = 1
    train_nn.X_DIM = img_scale.shape[1]
    train_nn.Y_DIM = img_scale.shape[0]

def main():
    img_scale, model  = parse_arguments()
    set_train_nn_global_variables(img_scale)
    print('\nX dimension:  {},    X range:  {}'.format(train_nn.X_DIM, train_nn.XRANGE))
    print("Y dimension:  {},    Y range:  {}\n".format(train_nn.Y_DIM, train_nn.YRANGE))

    ANT_LOC_LIST, Y = RunNN(4, train_nn.XRANGE, train_nn.YRANGE,img_scale, model)
    roomCoverage = get_stats(ANT_LOC_LIST, Y)
    best_loc_idx = np.argmax(roomCoverage)
    print("Best Antenna Location:  {}".format(ANT_LOC_LIST[best_loc_idx][:2]))
    print("Room Coverage Score:  {}\n".format(roomCoverage[best_loc_idx]))
    plot_signal_strength(Y[best_loc_idx, :, :], ANT_LOC_LIST[best_loc_idx])
    print("Success\n")

if __name__ == '__main__':
    main()
