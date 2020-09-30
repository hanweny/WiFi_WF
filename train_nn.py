import cv2, os, sys, time, pickle, warnings, argparse
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import pylab as pl
mpl.use('Agg')
from collections import defaultdict
from multiprocessing import Pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D,MaxPooling2D, MaxPooling1D, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import callbacks


##########################################################################
#####################   Feature Engineering  #############################
##########################################################################

def integrate_X_over_coeff(MAT, ANT_COOR):
    calc_distance = lambda x1, y1, x2, y2: np.sqrt((x1-x2)**2 + (y2-y1)**2)
    # x is the cartesian x axis; COL_ID represents columns
    # y is the cartesian y axis; ROW_ID represents rows
    LOC_X, LOC_Y = ANT_COOR # Y is inverted, original coordinate
    ROW_ID, COL_ID = int((YRANGE[1] - LOC_Y) // MESH_Y), int((LOC_X - XRANGE[0]) // MESH_X)
    MAT_NEW = MAT.copy()
    for i, y in enumerate(np.arange(YRANGE[1], YRANGE[0] - MESH_Y, -MESH_Y)):
        for j, x in enumerate(np.arange(XRANGE[0], XRANGE[1] + MESH_X, MESH_X)):
            power, scale = 0, np.sqrt(MESH_X**2 + MESH_Y**2)
            x1, x_intersect, x_idx, y1, y_intersect, y_idx = x, x, j, y, y, i
            ANT_X, ANT_COL_ID, ANT_Y, ANT_ROW_ID = LOC_X, COL_ID, LOC_Y, ROW_ID
            if LOC_X == x1:
                power = np.sum([MAT[row_idx, j] for row_idx in range(min(i, ROW_ID), max(i, ROW_ID) + 1)])
                MAT_NEW[i, j] = power * MESH_Y / scale
            else:
                k = (LOC_Y - y1) / (LOC_X - x1)
                b = LOC_Y - k * LOC_X
                if k < 0:
                    if x > LOC_X:
                        ANT_X, ANT_Y, ANT_ROW_ID, ANT_COL_ID, x1, x_intersect, x_idx, y1, y_intersect, y_idx = \
                            x1, y1, y_idx, x_idx, ANT_X, ANT_X, COL_ID, ANT_Y, ANT_Y, ROW_ID #switch the pos of ANT and point
                    while x1 != ANT_X or y1 != ANT_Y:
                        y_next = k * (x1 + MESH_X) + b
                        if y_next < y1 - MESH_Y:
                            x_next = ((y1 - MESH_Y) - b) / k
                            distance = np.sqrt((x_intersect - x_next)**2 + (y_intersect - (y1 - MESH_Y)) **2)
                            power += MAT[y_idx, x_idx] * distance / scale
                            x_intersect = x_next; y_intersect = y1 - MESH_Y; y1 = y1 - MESH_Y; y_idx += 1
                        elif y_next > y1 - MESH_Y:
                            distance = np.sqrt((x_intersect - (x1 + MESH_X))**2 + (y_intersect - y_next) **2)
                            power += MAT[y_idx, x_idx] * distance / scale
                            x_intersect = x1 + MESH_X; y_intersect = y_next; x1 = x1 + MESH_X; x_idx += 1
                        else:
                            power += MAT[y_idx, x_idx]
                            x1 += MESH_X; y1 -= MESH_Y; x_intersect = x1; y_intersect = y1; y_idx +=1; x_idx += 1
                    assert(y_idx == ANT_ROW_ID and x_idx == ANT_COL_ID)
                    distance = np.sqrt((x_intersect - ANT_X)**2 + (y_intersect - ANT_Y)**2)
                    power += MAT[ANT_ROW_ID, ANT_COL_ID] * distance / scale
                    MAT_NEW[i, j] = power
                elif k > 0:
                    if LOC_X < x:
                        ANT_X, ANT_Y, ANT_ROW_ID, ANT_COL_ID, x1, x_intersect, x_idx, y1, y_intersect, y_idx = \
                            x1, y1, y_idx, x_idx, ANT_X, ANT_X, COL_ID, ANT_Y, ANT_Y, ROW_ID #switch the pos of ANT and point
                    while x1 != ANT_X or y1 != ANT_Y:
                        y_next = k * (x1 + MESH_X) + b
                        if y_next < y1 + MESH_Y: #go right
                            distance = np.sqrt((x_intersect - (x1 + MESH_X))**2 + (y_intersect - y_next) **2)
                            power += MAT[y_idx, x_idx] * distance / scale
                            x_intersect = x1 + MESH_X; y_intersect = y_next; x1 = x1 + MESH_X; x_idx += 1
                        elif y_next > y1 + MESH_Y:
                            x_next = ((y1 + MESH_Y) - b) / k
                            distance = np.sqrt((x_intersect - x_next)**2 + (y_intersect - (y1 + MESH_Y)) **2)
                            power += MAT[y_idx, x_idx] * distance / scale
                            x_intersect = x_next; y_intersect = y1 + MESH_Y; y1 = y1 + MESH_Y; y_idx -= 1
                        else:
                            power += MAT[y_idx, x_idx]
                            x1 += MESH_X; y1 += MESH_Y; x_intersect = x1; y_intersect = y1; y_idx -=1; x_idx += 1
                    assert(y_idx == ANT_ROW_ID and x_idx == ANT_COL_ID)
                    distance = np.sqrt((x_intersect - ANT_X)**2 + (y_intersect - ANT_Y)**2)
                    power += MAT[ANT_ROW_ID, ANT_COL_ID] * distance / scale
                    MAT_NEW[i, j] = power
                else: #Horizontal Line
                    power = np.sum([MAT[i, col_idx] for col_idx in range(min(j, COL_ID), max(j, COL_ID) + 1)])
                    MAT_NEW[i, j] = power * MESH_X / scale
    return MAT_NEW


def construct_map(specs, geo_num):
    # 1 is metal, 2 is absorber, 3 is dielectric constant   
    #### Dielectric constant need to be manually selected!
    d_const = {1: "#00FF00", 2: "#0000FF", 3: "#FB0000"}
    fig = plt.figure(figsize = (10, 10), facecolor = "#FEFEFE")
    ax = plt.gca()
    plt.xlim(-8, 8)
    plt.ylim(-10, 10)
    plt.tick_params(width = 0.5)
    ax.set_axis_off()
    for s in specs:
        s = eval(s)
        if len(s) == 5: #Box
            r = plt.Rectangle((s[0], s[1]), height=s[2], width=s[3], color=d_const[s[4]], ec=d_const[s[4]])
            ax.add_artist(r)
        elif len(s) == 4: # Cylinder
            c = plt.Circle((s[0], s[1]), s[2], color=d_const[s[3]], ec = d_const[s[3]])
            ax.add_patch(c)
    plt.tight_layout(pad=0)
    plt.savefig('maps/out{}.png'.format(geo_num), facecolor=fig.get_facecolor(), dpi = 4.1, edgecolor='none')
    plt.close(fig)

    
def construct_X(geo_num, wifi_radius=1):
    def construct_X_helper(ANT_LOCATION, wifi_radius=1):
        x, y, z = ANT_LOCATION
        ###### FEATURE ENGINEERING STEP
        ## Map [0, 255] -> [0, 1/2] Take the negative part of logit, negate and then plus 1
        ## [dielectric constant, distance, angle]
        temp = np.zeros(shape = (int((YRANGE[1] - YRANGE[0]) / MESH_Y + 1), int((XRANGE[1] - XRANGE[0]) / MESH_X + 1), 4))
        for i, y1 in enumerate(np.arange(YRANGE[1], YRANGE[0] - MESH_Y, -MESH_Y)):
            for j, x1 in enumerate(np.arange(XRANGE[0], XRANGE[1] + MESH_X, MESH_X)):
                temp[i, j, 0] = -VAL_MAX if img_scale[i, j, 1] != 0 else 0 if img_scale[i, j, 2] != 0 else \
                                min(-np.log((img_scale[i, j, 0] / 255 / 2) / (1 - img_scale[i, j, 0] / 255 / 2)) + 1, VAL_MAX)
                temp[i, j, 2] = np.sqrt((x1-x)**2 + (y1-y)**2)
                temp[i, j, 3] = np.arctan((y1-y) / (x1 - x)) if x1-x != 0 else 3.14/2 if y1 > y else -3.14/2
        try:
            temp[:,:,1] = integrate_X_over_coeff(temp[:,:,0], [x, y])
        except Exception as e:
            raise Exception("Error: {}. Integration Failed".format(e))

        for r1 in np.arange(-wifi_radius, wifi_radius + 1):
            for r2 in np.arange(-wifi_radius, wifi_radius + 1):
                row, col = int((x - XRANGE[0]) * scale) + r2, int((YRANGE[1] - y) * scale) + r1
                if row >= 0 and row <= (XRANGE[1] - XRANGE[0]) * scale and \
                   col >= 0 and col <= (YRANGE[1] - YRANGE[0]) * scale:
                    temp[col, row] = [-1, 5, 0, -1]
        return temp

    spec_img_path = 'maps/out{}.png'.format(geo_num) if type(GEO) is not str else \
                    os.path.join(GEO, 'out{}.png'.format(geo_num))
    
    spec_img_path = GEO if SINGLE_GEO and os.path.isfile(GEO) else os.path.join(GEO, 'out1.png') if SINGLE_GEO else spec_img_path
    
    img_scale = cv2.cvtColor(cv2.imread(spec_img_path), cv2.COLOR_BGR2RGB)
    X = []; scale = int(1 // MESH)
    if SINGLE_GEO:
        X.append(construct_X_helper(master[geo_num]['LOCATION'], wifi_radius))
    else:
        for k in master[geo_num]:
            X.append(construct_X_helper(master[geo_num][k]['LOCATION'], wifi_radius))
    return X

def construct_y(geo_num):
    y = []
    if SINGLE_GEO:
        y.append(master[geo_num]['df'].pivot(index = 'Y', columns = 'X', values = ['Etot']).iloc[::-1].values)
    else:
        for k in master[geo_num]:
            y.append(master[geo_num][k]["df"].pivot(index = 'Y', columns = 'X', values = ['Etot']).iloc[::-1].values)
    return y

def my_func(geo_num):
    if type(GEO) is not str:
        construct_map(GEO.loc[geo_num], geo_num)
    X = construct_X(geo_num, 1)
    y = construct_y(geo_num)
    return geo_num, X, y


def construct_X_y(GEO_NUM_LIST):
    global master
    X, y = {}, {}
    print("Start Preprocessing Data...")
    if SINGLE_GEO:
        master_temp = master[list(master.keys())[-1]]
        master = master_temp
        GEO_NUM_LIST = list(master.keys())

    pool = Pool(processes=min(8, (len(GEO_NUM_LIST) + 1) // 2))
    count, processed = 0, 0
    for geo_num, X_temp, y_temp in pool.imap_unordered(my_func, GEO_NUM_LIST):
        X[geo_num] = np.array(X_temp)
        y[geo_num] = np.array(y_temp)
        count += 1
        if count / len(GEO_NUM_LIST) >= processed:
            print("{:.1f}% processsed".format(processed*100))
            processed += 0.2
    if SINGLE_GEO:
        X = np.concatenate([v for k, v in sorted(X.items(), key = lambda x: x[0])])
        y = np.concatenate([v for k, v in sorted(y.items(), key = lambda x: x[0])])
    pool.close()
    pool.join()
    print("Data Preprocessed")
    return X, y

def model_train_test_split(X, y):
    np.random.seed(48)
    if SINGLE_GEO:
        TRAIN_IDX = np.random.choice(list(range(X.shape[0])), int(X.shape[0] * 0.8), replace = False)
        TEST_IDX = set(list(range(X.shape[0])))- set(TRAIN_IDX)
    else:
        TRAIN_IDX = np.random.choice(list(master.keys()), int(len(master) * 0.8), replace = False)
        TEST_IDX = set(list(master.keys()))- set(TRAIN_IDX)
    X_train = np.array([X[i] for i in TRAIN_IDX]).reshape(-1, Y_DIM, X_DIM, 4)
    y_train = np.array([y[i] for i in TRAIN_IDX]).reshape(-1, Y_DIM, X_DIM)
    X_test = np.array([X[i] for i in TEST_IDX]).reshape(-1, Y_DIM, X_DIM, 4)
    y_test = np.array([y[i] for i in TEST_IDX]).reshape(-1, Y_DIM, X_DIM)
    print("Training Size:  {}  Testing Size:   {}".format(X_train.shape[0], X_test.shape[0]))
    return X_train, X_test, y_train, y_test, TRAIN_IDX, TEST_IDX


##########################################################################
#####################   Feature Engineering  #############################
##########################################################################

def examine_xy(model_current, TEST_IDX, X_test, y_test): 
    TEST_IDX = [i for i in TEST_IDX]
    np.random.seed(int(time.time()))
    df_idx = np.random.choice(TEST_IDX, 1) if SINGLE_GEO else np.random.choice(list(master[TEST_IDX[0]].keys()), 1)
    df = master[df_idx[0]] if SINGLE_GEO else master[TEST_IDX[0]][df_idx[0]]
    idx = 0
    if SINGLE_GEO:
        while df_idx != TEST_IDX[idx]:
            idx += 1
    else:
        while df_idx != list(master[TEST_IDX[0]].keys())[idx]:
            idx += 1
                
    ## Y Comparisons
    plt.subplots(2, 3, figsize = (16, 8))

    ax = plt.subplot(2, 3, 1)
    img1 = ax.matshow(df["df"].pivot(index = 'Y', columns = 'X', values = ['Etot']).iloc[::-1].values)
    pl.colorbar(img1)
    plt.title("Actual Y\n", fontsize = 16)

    ax = plt.subplot(2, 3, 2)
    img2 = ax.matshow(y_test[idx, :,:])
    pl.colorbar(img2)
    plt.title("Model Y\n", fontsize = 16)
    
    ax = plt.subplot(2, 3, 3)
    img3 = ax.matshow(model_current.predict(X_test[idx,].reshape(-1, Y_DIM, X_DIM, 4))[0])
    pl.colorbar(img3)
    plt.title("Pred Y\n", fontsize = 16)
                      
    ax = plt.subplot(2, 4, 5)
    img4 = ax.matshow(X_test[idx,:,:,0])
    pl.colorbar(img4)
    plt.title("Dielectric Layer\n", fontsize = 16)


    ax = plt.subplot(2, 4, 6)
    img5 = ax.matshow(X_test[idx,:,:,1])
    pl.colorbar(img5)
    plt.title("Integration Layer\n", fontsize = 16)

    ax = plt.subplot(2, 4, 7)
    img6 = ax.matshow(X_test[idx,:,:,2])
    pl.colorbar(img6)
    plt.title("Distance Layer\n", fontsize = 16)

    ax = plt.subplot(2, 4, 8)
    img7 = ax.matshow(X_test[idx,:,:,3])
    pl.colorbar(img7)
    plt.title("Angle Layer\n", fontsize = 16)
    
    plt.suptitle("ANT LOC:  {}".format(df["LOCATION"]), fontsize = 18)

   
    plt.savefig("examined_output.png")

##########################################################################
#######################   Model Specifics  ###############################
##########################################################################

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()))

def build_model(args):
    model_shape_matches = False if model is None else \
                          all([model.layers[0].input_shape[1] == Y_DIM, model.layers[0].input_shape[2] == X_DIM])
    # If given model matches current shape and no transfer learning flags 
    if model is not None and not args.transfer and model_shape_matches:
        return None
    
    print("Building Model\n")
    model_current = Sequential()
    model_current.add(BatchNormalization(input_shape=(Y_DIM, X_DIM, 4)))
    model_current.add(Conv2D(8, (2, 2), activation='relu'))
    model_current.add(Conv2D(16, (2, 2), activation='relu'))
    model_current.add(MaxPooling2D(pool_size=(2,2)))
    model_current.add(Flatten())
    model_current.add(Dense(5000, activation = 'relu'))
    model_current.add(Dropout(0.3))
    model_current.add(Dense(Y_DIM * X_DIM, kernel_initializer='normal'))
    model_current.add(Reshape((Y_DIM, X_DIM)))
    model_current.compile(loss='mean_squared_error', optimizer='adam', metrics = [coeff_determination, 'mse'])
    model_current.summary()
    ## Model is given but input shape does not match or transfer learning flag is set
    if model is not None and (not model_shape_matches or args.transfer):
        print("\nUsing Transfer learning\n")
        try:
            for new_layer, layer in zip(model_current.layers[1:-4], model.layers[1:-4]):
                new_layer.set_weights(layer.get_weights())
        except:
            raise Exception("Error when transfering weights")
    return model_current


def train_model(model_current, X_train, y_train):
    print("\nStart Training Model...\n")
    es = callbacks.EarlyStopping(monitor="val_mean_squared_error", min_delta = 0.001, baseline = 0.3, mode="min", verbose=1, patience=10)
    history = model_current.fit(x=X_train, y=y_train, batch_size=3, epochs=10, verbose=1, validation_split=0.2, shuffle=True, callbacks = [es])
    return history


def print_model_metrics(model_current, X_test, y_test, history):
    print("Testing Scores: {}".format(list(np.round(model_current.evaluate(X_test, y_test)[1:], decimals=4))))
    print("val_RMSE: {}".format(list(np.round(np.sqrt(history.history["val_mean_squared_error"]), decimals=4))))
    print("val_Rsquared: {}".format(np.round(history.history["val_coeff_determination"], decimals=4)))

    
    
##########################################################################
##################  Argument Parser & Global Varibales ###################
##########################################################################    

master, GEO, model, VAL_MAX, PIXEL_MARGIN_ERROR_THRESHOLD, XRANGE, YRANGE, MESH_X, MESH_Y, MESH, X_DIM, Y_DIM, SINGLE_GEO = [None for i in range(13)]

def set_global_variables(master):
    global VAL_MAX, PIXEL_MARGIN_ERROR_THRESHOLD, XRANGE, YRANGE, MESH_X, MESH_Y, MESH, X_DIM, Y_DIM
    VAL_MAX = 10**2
    PIXEL_MARGIN_ERROR_THRESHOLD = 100  ## ManuallY specified should be FF at the B and G channel
    sample = list(list(master.values())[-1].values())[-1]['df']
    XRANGE = (min(sample['X']), max(sample['X']))
    YRANGE = (min(sample['Y']), max(sample['Y']))
    MESH_X = sorted(np.unique(sample['X']))[1] - XRANGE[0]
    MESH_Y = sorted(np.unique(sample['Y']))[1] - YRANGE[0]
    assert(MESH_X == MESH_Y)
    MESH = MESH_X
    X_DIM = int((XRANGE[1] - XRANGE[0] + MESH) // MESH)
    Y_DIM = int((YRANGE[1] - YRANGE[0] + MESH) // MESH)
    assert(X_DIM == len(np.unique(sample['X'])))
    assert(Y_DIM == len(np.unique(sample['Y'])))
    
    
def parse_arguments():
    global master, GEO, SINGLE_GEO, model
    # argument parser
    parser = argparse.ArgumentParser(description='Simulation Training supports transfer learning. (Four cases: 1) Model is not given -> train a new model. 2) Model is given: a) input shape does not match current geometry dimension -> transfer learning; b) input shape matches current geometry dimension -> transfer learning (if --transfer flag is given) or simply return.')
    
    parser.add_argument('data_path', type=str, 
                        help='Preprocessed data pickle file path')
    parser.add_argument('geo_path', type=str,
                       help = 'Support three modes: 1) Path to geometry csv specs file. 2) Dir to spec images files (image filenames in Dir have to match "out{geo_num}.png" format. 3) Path to a single image file (if there is only one geometry in the given data pickle file.)')
    parser.add_argument('-m', '--model', type=str,
                       help = 'Path of trained model. (Special case: If model input shape matches given data geometry dimension and transfer learning flag is not set, then simply return.)')
    parser.add_argument('-t', '--transfer', action='store_true',
                       help = 'Flag to use transfer learning. Model path must be provided.')  # Use 'store_true' as a flag'
    parser.add_argument('-e', '--examine', action='store_true', 
                       help = 'Flag to output one featured engineered and prediction results.')
    parser.add_argument('-o', '--output_path', type=str, default = './trained_nn_weights.h5', 
                        help = 'Alternative trained model saving path (Default is ./trained_nn_weights.h5)')
    args = parser.parse_args()

    ## load data from params
    if not os.path.exists(args.data_path):
        raise Exception("Preprocessed data pickle does not exists at: {}".format(args.data_path))
    if not os.path.exists(args.geo_path):
        raise Exception("Geolocation file or preprocessed image does not exist at : {}".format(args.geo_path)) 
    try:
        master = pickle.load(open(args.data_path, 'rb'))
    except:
        raise Exception("Eroor when loading data pkl file at: '{}'".format(args.data_path))
        
    try:
        if len(master) == 1:
            SINGLE_GEO = True
           
        if args.geo_path[-3:] == 'csv':
            GEO = pd.read_csv(args.geo_path)
            if not os.path.exists('./maps'):
                os.makedirs('maps')
        else:
            assert(os.path.isdir(args.geo_path) or (os.path.isfile(args.geo_path) and SINGLE_GEO))
            GEO = args.geo_path
    except:
        raise Exception("Unsupported file format: {}".format(args.geo_path.split('/')[-1]))
    
    model = None
    if args.model and not os.path.exists(args.model):
        print("\nALERT: Given model does not exists at: '{}'. WILL TRAIN FROM SCRATCH.\n".format(args.model))
        #raise Exception("Model does not exists at: '{}'".format(args.model))
    if args.model and os.path.exists(args.model):
        try:
            model = load_model(args.model, custom_objects = {"coeff_determination": coeff_determination})
        except Exceptiona as e:
            print("Error:  ", e)
            print("Cannot Load the nn weights")
    return args    


##########################################################################
##########################  Main Function ################################
##########################################################################

def main():
    args = parse_arguments()
    set_global_variables(master)
    model_current = build_model(args)
    if model_current is None:
        print("Given model has same Input Size. No transfer learning flag detected. Exiting...")
        return
    X, y = construct_X_y(master.keys())
    X_train, X_test, y_train, y_test, TRAIN_IDX, TEST_IDX = model_train_test_split(X, y)
    history = train_model(model_current, X_train, y_train)
    print("\nModel Trained\n")
    print_model_metrics(model_current, X_test, y_test, history)
            
    print("\nSaving Model...")
    model_current.save(args.output_path)
    
    if args.examine:
        print("\nGenerating examining XY plots...")
        examine_xy(model_current, TEST_IDX, X_test, y_test)
        
    print("\nDONE") 
    

if __name__ == '__main__':
    main()
    
    
