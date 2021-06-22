from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools
from drawGraph import draws

# split
# python create_split.py -d datasets/eccv16_dataset_summe_google_pool5.h5 --save-dir datasets --save-name summe_splits  --num-splits 5

# python create_split.py -d datasets/eccv16_dataset_youtube_google_pool5.h5 --save-dir datasets --save-name youtube_splits  --num-splits 10


# train 명령어
# -m의 tvsum : avg , summe : max 방식
# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --verbose
# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split-CNN_LSTM --split-id 0 --verbose

# layer
# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --num-layers 3 --gpu 0 --save-dir log/summe-split0-GRU3layer --split-id 0 --verbose
# python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s datasets/tvsum_splits.json -m tvsum --num-layers 3 --gpu 0 --save-dir log/tvsum-split0-GRU3lyaer --split-id 0 --verbose

# hidden dim (default = 256)
# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --hidden-dim 1024 --gpu 0 --save-dir log/summe-split0-LSTM1024hd --split-id 0 --verbose
# python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s datasets/tvsum_splits.json -m tvsum --hidden-dim 1024 --gpu 0 --save-dir log/tvsum-split0-LSTM1024hd --split-id 0 --verbose


# python main.py -d datasets/eccv16_dataset_ovp_google_pool5.h5 -s datasets/ovp_splits.json -m tvsum --num-layers 1 --gpu 0 --save-dir log/ovg-split0-LSTM1lyaer --split-id 0 --verbose


# test 명령어
# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --evaluate --resume path_to_your_model.pth.tar --verbose --save-results

# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0-LSTM1layer --split-id 0 --evaluate --resume log/summe-split0-LSTM1layer/model_epoch60.pth.tar --verbose --save-results
# python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s datasets/tvsum_splits.json -m summe --gpu 0 --save-dir log/summe-split1 --split-id 1 --evaluate --resume log/summe-split1/model_epoch60.pth.tar --verbose --save-results


# python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s datasets/tvsum_splits.json -m tvsum --gpu 0 --save-dir log/tvsum-split0-LSTM1Lyaer --split-id 0 --evaluate --resume log/tvsum-split0-LSTM1Lyaer/model_epoch60.pth.tar --verbose --save-results

# python visualize_results.py -p log/summe-split0-LSTM1layer/result.h5
# python visualize_results.py -p log/summe-split1/result.h5
# python summary2video.py -p log/summe-split0/result.h5 -d datasets/test01 -i 3 --fps 30 --save-dir log --save-name summary.mp4
# python summary2video.py -p log/summe-split1/result.h5 -d datasets/test01 -i 3 --fps 30 --save-dir log --save-name summary.mp4

# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0-LSTM2layer --split-id 0 --evaluate --resume log/summe-split0-LSTM2layer/model_epoch60.pth.tar --verbose --save-results

# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --num-layers 2 --gpu 0 --save-dir log/summe-split0-LSTM2layer --split-id 0 --verbose



# 최종 발표 시연 (GRU로 하려면 모델 바꿔주기)
# 1. train - 각각 summe LSTM 1 layer / tvsum LSTM 1 layer
# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --num-layers 1 --gpu 0 --save-dir log/summe-split0-LSTM1layer --split-id 0 --verbose
# python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s datasets/tvsum_splits.json -m tvsum --num-layers 1 --gpu 0 --save-dir log/tvsum-split0-LSTM1lyaer --split-id 0 --verbose

# 2. test - 각각 summe LSTM 1 layer / tvsum LSTM 1 layer
# python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/summe-split0-LSTM1layer --split-id 0 --evaluate --resume log/summe-split0-LSTM1layer/model_epoch60.pth.tar --verbose --save-results
# python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s datasets/tvsum_splits.json -m tvsum --gpu 0 --save-dir log/tvsum-split0-LSTM1Lyaer --split-id 0 --evaluate --resume log/tvsum-split0-LSTM1Lyaer/model_epoch60.pth.tar --verbose --save-results


parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
parser.add_argument('-s', '--split', type=str, required=True, help="path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'],
                    help="evaluation metric ['tvsum', 'summe']")
# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=60, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")

parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

def main():
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    splits = read_json(args.split)
    assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
    split = splits[args.split_id]
    train_keys = split['train_keys']
    test_keys = split['test_keys']
    print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

    # model : DSNetAF , DSNet, DSN, LSTM
    print("Initialize model")
    #model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    model = LSTM(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers)

    # base_model : linear, lstm, bilstm, gcn, attention
    #model = DSNet(base_model = 'linear', num_feature=args.input_dim, num_hidden=args.hidden_dim, anchor_scales, num_head)
    #model = DSNetAF(base_model = 'linear', num_feature=args.input_dim, num_hidden=args.hidden_dim, num_head)

    #model = CNN_LSTM(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers)


    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=False)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        evaluate(model, dataset, test_keys, use_gpu)
        return

    print("==> Start training")
    start_time = time.time()
    model.train()
    baselines = {key: 0. for key in train_keys} # baseline rewards for videos
    reward_writers = {key: [] for key in train_keys} # record reward changes for each video



    for epoch in range(start_epoch, args.max_epoch):
        idxs = np.arange(len(train_keys))   # training video의 길이
        np.random.shuffle(idxs) # shuffle indices

        for idx in idxs:
            key = train_keys[idx]       # = training video 숫자
            seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
            seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
            if use_gpu: seq = seq.cuda()

            probs = model(seq) # output shape (1, seq_len, 1)   # model (baseline = LSTM) 돌려서 나온 Q값

            cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
            m = Bernoulli(probs)
            epis_rewards = []

            for _ in range(20): #range(args.num_episode):
                # 랜덤하게 샘플 생성
                actions = m.sample()
                # actions으로 계산된 확률 밀도/질량 함수의 로그 반환
                log_probs = m.log_prob(actions)

                reward = compute_reward(seq, actions, use_gpu=use_gpu)
                expected_reward = log_probs.mean() * (reward - baselines[key])
                cost -= expected_reward # minimize negative expected reward
                epis_rewards.append(reward.item())

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            reward_writers[key].append(np.mean(epis_rewards))

        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
        print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))

    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))
    evaluate(model, dataset, test_keys, use_gpu)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
    model_save_path = osp.join(args.save_dir, 'model_epoch' + str(args.max_epoch) + '.pth.tar')
    save_checkpoint(model_state_dict, model_save_path)
    print("Model saved to {}".format(model_save_path))

    dataset.close()


def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")
    titleNum = 0
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'        # tvsum : avg / summe : max
        #eval_metric = 'max'
        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]

            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]
            n_steps = dataset[key]['n_steps'][...]          # 15프레임마다 나눴을때 개수

            #print("k: ", key)
            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            # args.save_dir  # log/summe-split0-GRU1layer
            titleName = args.save_dir + "/graph_" + str(key)
            #print(titleName)
            draws(titleName, num_frames, probs, machine_summary, cps, positions, n_steps)
            titleNum += 1

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm

if __name__ == '__main__':
    main()
