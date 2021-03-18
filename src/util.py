import numpy as np
import yaml

#calculate training time
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"

#save array to txt file
def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                for k in range(results.shape[2]):
                    f.write('{:.3f} '.format(results[i, j, k]))
            f.write('\n')

def get_nyu_dataset():
    with open("../configs/config.yaml", "r+", encoding='utf8') as fo:
        txt = fo.read()
        return txt.split(' ')[1]

def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
        positions = np.array(positions).reshape(8252, 14, 3)
    return positions


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu', 'msra'])


def get_dataset_file(dataset):
    return '../result/{}_test_groundtruth_label.txt'.format(dataset)


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.036865, -587.075073, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = - x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


def get_errors(dataset, in_file):
    # if not check_dataset(dataset):
    #     print('invalid dataset: {}'.format(dataset))
    #     exit(-1)
    labels = get_positions(get_dataset_file(dataset))
    outputs = get_positions(in_file)
    params = get_param(dataset)
    labels = pixel2world(labels, *params)
    outputs = pixel2world(outputs, *params)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return errors

def get_msra_viewpoint(in_file):
    with open(in_file) as f:
        viewpoint = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(viewpoint), (-1, 2))












