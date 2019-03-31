import os
import numpy as np
import pickle

ORIGINAL_DATA_PATH = 'data/original'
INPUT_DATA_PATH = 'data/input'
ACTION_NUM, SUBJECT_NUM, EPOCH_NUM = 20, 10, 3
JOINT_NUM = 20


def parse_each_file(lines):
    skeleton_coords = []

    for i in range(len(lines) // JOINT_NUM):
        frame_data = np.empty((20, 3), dtype=np.float32)
        frame_txt = lines[i * JOINT_NUM:i * JOINT_NUM + JOINT_NUM]
        for j, l in enumerate(frame_txt):
            frame_data[j] = np.array([float(f) for f in l.split()[:3]])
        skeleton_coords.append(frame_data)

    return np.array(skeleton_coords)


if __name__ == "__main__":
    dataset = []

    for a in range(1, ACTION_NUM + 1):
        for s in range(1, SUBJECT_NUM + 1):
            for e in range(1, EPOCH_NUM + 1):
                filename = os.path.join(
                    ORIGINAL_DATA_PATH, "a%02d_s%02d_e%02d_skeleton.txt" % (a, s, e))
                try:
                    f = open(filename, 'r')
                    lines = f.readlines()
                    assert len(lines) % JOINT_NUM == 0, 'file format error'
                    dataset.append((a, parse_each_file(lines)))
                except FileNotFoundError as e:
                    continue

    pickle.dump(dataset, open(os.path.join(
        INPUT_DATA_PATH, 'input.pkl'), 'wb'))
