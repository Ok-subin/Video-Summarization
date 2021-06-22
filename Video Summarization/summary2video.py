import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to h5 result file")
#parser.add_argument('-d', '--frm-dir', type=str, required=True, help="path to frame directory")
parser.add_argument('-d', '--frm-dir', type=str, help="path to frame directory")
parser.add_argument('-i', '--idx', type=int, default=0, help="which key to choose")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")
parser.add_argument('--save-dir', type=str, default='log', help="directory to save")
parser.add_argument('--save-name', type=str, default='summary.mp4', help="video name to save (ends with .mp4)")
args = parser.parse_args()

# summe
# python summary2video.py -p log/summe-split0-LSTM1layer/result.h5 -d datasets/summe5 -i 1 --fps 30 --save-dir log/summe-split0-LSTM1layer --save-name summary_idx1.mp4
# python summary2video.py -p log/summe-split0-LSTM1layer/result.h5 -d datasets/"video 위치!!!!" -i 1 --fps 30 --save-dir log/summe-split0-LSTM1layer --save-name summary_idx1.mp4

# python summary2video.py -p log/summe-split0-LSTM1layer/result.h5 -d datasets/summe/ -i 1 --fps 30 --save-dir log/summe-split0-LSTM1layer

# tvsum
# python summary2video.py -p log/summe-split1/result.h5 -d datasets/tvsum9 -i 9 --fps 30 --save-dir log --save-name summary.mp4




# 최종 발표 시연
# 1. summe - LSTM 1 layer
# python summary2video.py -p log/summe-split0-LSTM1layer/result.h5 -d datasets/summe/ -i 4 --fps 30 --save-dir log/summe-split0-LSTM1layer

# 2. tvsum - LSTM 1 layer
# python summary2video.py -p log/tvsum-split0-LSTM1layer/result.h5 -d datasets/tvsum/ -i 9 --fps 30 --save-dir log/tvsum-split0-LSTM1layer


def frm2video(frm_dir, summary, vid_writer):
    for idx, val in enumerate(summary):
        if val == 1:
            #print(idx)
            # here frame name starts with '000001.jpg'
            # change according to your need
            # frm_name = str(idx+1).zfill(6) + '.jpg'
            frm_name = str(idx+1) + '.jpg'
            frm_path = osp.join(frm_dir, frm_name)
            frm = cv2.imread(frm_path)
            frm = cv2.resize(frm, (args.width, args.height))
            vid_writer.write(frm)


if __name__ == '__main__':
    if not osp.exists(args.save_dir):
        os.mkdir(args.save_dir)

    h5_res = h5py.File(args.path, 'r')
    key = list(h5_res.keys())[args.idx]

    save_name = "summary_" + str(key) + ".mp4"

    vid_writer = cv2.VideoWriter(
        osp.join(args.save_dir, save_name),
       # cv2.VideoWriter_fourcc(*'MP4V'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        (args.width, args.height),
    )

    #print(args.fps)


    print("해당 idx의 video :", key)
    summary = h5_res[key]['machine_summary'][...]    # ex) [1. 1. 1. ... 0. 0. 0.]

    h5_res.close()
    frm_dir = str(args.frm_dir) + key
    #print(frm_dir)
    #print(frm_dir)
    #frm2video(args.frm_dir, summary, vid_writer)
    frm2video(frm_dir, summary, vid_writer)
    vid_writer.release()