import os
import numpy as np

import torch.nn.functional as F
import utils
from train2 import VideoSSL
from video_dataset import VideoDataset
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from evals import phase_classification
from evals.kendalls_tau import evaluate_kendalls_tau

import random
import argparse
import glob
from natsort import natsorted

def get_embeddings(model, data_loader, args):

    embeddings = []
    labels = []
    frame_paths = []
    names = []
    masks = []

    # device = f"cuda:{args.device}"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(model.device)
    # model.to(device)
    device = model.device
    model.eval()
    # print(device)
    with torch.no_grad():
        for batch in data_loader:
            # print(act_iter)
            # print(len(act_iter))
            # for (features_X, mask_X, gt_X, video_fname_X, unique_actions_X), (features_Y, mask_Y, gt_Y, video_fname_Y, unique_actions_Y) in act_iter:
            # features_X, mask_X, gt_X, video_fname_X, n_subactions_X = act_iter[0]
            # features_Y, mask_Y, gt_Y, video_fname_Y, n_subactions_Y = act_iter[1]
            # print(batch)
            features_raw_X, mask_X, gt_X, video_fname_X, n_subactions_X = batch[0]
            features_raw_Y, mask_Y, gt_Y, video_fname_Y, n_subactions_Y = batch[1]

            features_raw_X = features_raw_X.to(device=device)
            features_raw_Y = features_raw_Y.to(device=device)
            # print(device)
            # print(features_raw_X.device)
            # print(features_raw_X)
            # MLP's last layer size (D=40 for penn_action)
            D = model.layer_sizes[-1]

            ## Process the features of video X
            # B(batch size), T(no. of frames or timesteps) and _(feature dim/frame embedding size, which is 1024 in penn_action)
            B_X, T_X, _ = features_raw_X.shape
            # Reshape features_raw from 3D(B x T x _) to 2D(B*T x _) tensor, while keeping the feature dim intact
            # Pass it through the MLP: input layer size=_ and output layer size=D
            # Reshape the MLP output from (B*T x D) to (B x T x D)
            # Normalize features along the last dimension(-1), so each feature vector along D and across B and T has unit norm, for stable training
            embeddings_X = F.normalize(
                model.mlp(features_raw_X.reshape(-1, features_raw_X.shape[-1])).reshape(B_X, T_X, D),
                dim=-1
            )

            ## Process the features of video Y
            B_Y, T_Y, _ = features_raw_Y.shape
            embeddings_Y = F.normalize(
                model.mlp(features_raw_Y.reshape(-1, features_raw_Y.shape[-1])).reshape(B_Y, T_Y, D),
                dim=-1
            )

            # features_X = features_X.to(device).unsqueeze(0)
            # features_Y = features_Y.to(device).unsqueeze(0)
            
            # original_X = features_X.shape[1] // 2
            # original_Y = features_Y.shape[1] // 2
            
            # print(features_X.shape)

            # features_X = features_X[:, :original_X, :, :]
            # features_Y = features_Y[:, :original_Y, :, :]
            
            # b_X = features_X[:, -1].clone()
            # b_Y = features_Y[:, -1].clone()

            # b_X = torch.stack([b_X] * ((args.num_frames * 2) - features_X.shape[1]), axis=1).to(device)
            # b_Y = torch.stack([b_Y] * ((args.num_frames * 2) - features_Y.shape[1]), axis=1).to(device)
            
            # features_X = torch.cat([features_X, b_X], axis=1)
            # features_Y = torch.cat([features_Y, b_Y], axis=1)
            
            # embeddings_X = model(features_X)[:, :original_X, :]
            # embeddings_Y = model(features_Y)[:, :original_Y, :]
            
            # if args.verbose:
            #     print(f'Embedding shapes - X: {embeddings_X.shape}, Y: {embeddings_Y.shape}')
            # print(embeddings_X.shape)
            # print(embeddings_X.shape)
            embeddings_X = embeddings_X.squeeze(0).detach().cpu().numpy()
            embeddings_Y = embeddings_Y.squeeze(0).detach().cpu().numpy()

            # print(embeddings_X.shape)
            name_X = str(video_fname_X[0])
            name_Y = str(video_fname_Y[0])
            
            
            # print(gt_X.shape)
            # print(gt_X)

            lab_X = gt_X.detach().cpu().numpy().flatten()
            lab_Y = gt_Y.detach().cpu().numpy().flatten()
            
            # print(lab_X.shape)
            # print(gt_X.shape)
            # end_X = min(embeddings_X.shape[0], len(lab_X))
            # end_Y = min(embeddings_Y.shape[0], len(lab_Y))
            
            # print(len(lab_X))
            # print(embeddings_X.shape[0])
            # if embeddings_X.shape[1] != lab_X.shape[1]:
            #     print("Shape mismatch")
            #     print(embeddings_X.shape)
            #     print(lab_X.shape)      

            mask_X = mask_X.detach().cpu().numpy()
            mask_Y = mask_Y.detach().cpu().numpy()
            
            embeddings.append(embeddings_X)
            embeddings.append(embeddings_Y)
            
            labels.append(lab_X)
            labels.append(lab_Y)
            

            frame_paths.append(video_fname_X[0])
            frame_paths.append(video_fname_Y[0])
            
            names.append(name_X)
            names.append(name_Y)

            masks.append(mask_X)
            masks.append(mask_Y)
            # print(names)
            # print(labels)
            # print(embeddings)
    return embeddings, names, labels, masks

def main(ckpts, args):
    summary_dest = os.path.join(args.dest, 'eval_logs')
    os.makedirs(summary_dest, exist_ok=True)

    for ckpt in ckpts:
        writer = SummaryWriter(summary_dest, filename_suffix='eval_logs')

        # get ckpt-step from the ckpt name
        # _, ckpt_step = ckpt.split('.')[0].split('_')[-2:]
        # ckpt_step = int(ckpt_step.split('=')[1])
        ckpt_step = 1
        DEST = os.path.join(args.dest, 'eval_step_{}'.format(ckpt_step))

        device = f"cuda:{args.device}"
        model = VideoSSL.load_from_checkpoint(ckpt, map_location=device)
     
        model.eval()

        # grad off
        torch.set_grad_enabled(False)

        # if args.num_frames:
        #     CONFIG.TRAIN.NUM_FRAMES = args.num_frames
        #     CONFIG.EVAL.NUM_FRAMES = args.num_frames

        # CONFIG.update(model.hparams.config)

        # print(model.hparams)
        data_path = args.data_path

        # Create dataset and data loader
        train_dataset = VideoDataset('../data', args.dataset, args.n_frames, standardise=args.std_feats, split='train', random=True, action_class=args.activity)
        val_dataset = VideoDataset('../data', args.dataset, args.n_frames, standardise=args.std_feats, split='val', random=False, action_class=args.activity)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        all_classifications = []
        all_kendalls_taus = []
        # all_phase_progressions = []
        ap5, ap10, ap15 = 0, 0, 0

        for i_action in range(train_dataset.n_subactions):
            # Set action sequence if necessary, otherwise skip this part
            train_act_name = val_act_name = train_dataset.get_action_name(i_action)

            val_embs, val_names, val_labels, val_mask = get_embeddings(model, val_loader, args)
            train_embs, train_names, train_labels, train_mask = val_embs, val_names, val_labels, val_mask

            os.makedirs(DEST, exist_ok=True)
            DEST_TRAIN = os.path.join(DEST, f'train_{train_act_name}_embs.npy')
            DEST_VAL = os.path.join(DEST, f'val_{val_act_name}_embs.npy')

            np.save(DEST_TRAIN, {'embs': train_embs, 'names': train_names, 'labels': train_labels})
            np.save(DEST_VAL, {'embs': val_embs, 'names': val_names, 'labels': val_labels})

            train_embeddings = np.load(DEST_TRAIN, allow_pickle=True).tolist()
            val_embeddings = np.load(DEST_VAL, allow_pickle=True).tolist()

            train_embs = train_embeddings['embs']
            train_labels = train_embeddings['labels']
            train_names = train_embeddings['names']
            val_embs = val_embeddings['embs']
            val_labels = val_embeddings['labels']
            val_names = val_embeddings['names']

            # Evaluating Classification
            train_acc, val_acc = phase_classification.evaluate_phase_classification(ckpt_step, train_embs, train_labels, val_embs, val_labels, act_name=train_act_name, CLASSIFICATION_FRACTIONS=[0.1, 0.5, 1.0], writer=writer, verbose=args.verbose)
            ap5, ap10, ap15 = phase_classification.compute_ap(val_embs, val_labels)

            all_classifications.append([train_acc, val_acc])

            # Evaluating Kendall's Tau
            train_tau, val_tau = evaluate_kendalls_tau(train_embs, val_embs, stride=args.stride, kt_dist='sqeuclidean', visualize=False)
            all_kendalls_taus.append([train_tau, val_tau])

            print(f"Kendall's Tau: Stride = {args.stride}")
            print(f"Train = {train_tau}")
            print(f"Val = {val_tau}")

            writer.add_scalar(f'kendalls_tau/train_{train_act_name}', train_tau, global_step=ckpt_step)
            writer.add_scalar(f'kendalls_tau/val_{val_act_name}', val_tau, global_step=ckpt_step)

        train_classification, val_classification = np.mean(all_classifications, axis=0)
        train_kendalls_tau, val_kendalls_tau = np.mean(all_kendalls_taus, axis=0)

        writer.add_scalar('metrics/AP@5_val', ap5, global_step=ckpt_step)
        writer.add_scalar('metrics/AP@10_val', ap10, global_step=ckpt_step)
        writer.add_scalar('metrics/AP@15_val', ap15, global_step=ckpt_step)
        writer.add_scalar('metrics/all_classification_train', train_classification, global_step=ckpt_step)
        writer.add_scalar('metrics/all_classification_val', val_classification, global_step=ckpt_step)
        writer.add_scalar('metrics/all_kendalls_tau_train', train_kendalls_tau, global_step=ckpt_step)
        writer.add_scalar('metrics/all_kendalls_tau_val', val_kendalls_tau, global_step=ckpt_step)

        print(f'metrics/AP@5_val {ap5} global_step={ckpt_step}')
        print(f'metrics/AP@10_val {ap10} global_step={ckpt_step}')
        print(f'metrics/AP@15_val {ap15} global_step={ckpt_step}')
        print(f'metrics/all_classification_train {train_classification} global_step={ckpt_step}')
        print(f'metrics/all_classification_val {val_classification} global_step={ckpt_step}')
        print(f'metrics/all_kendalls_tau_train {train_kendalls_tau} global_step={ckpt_step}')
        print(f'metrics/all_kendalls_tau_val {val_kendalls_tau} global_step={ckpt_step}')

        writer.flush()
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, default=None) #default='./checkpoints')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dest', type=str, default='./log')

    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='Cuda device to be used')
    parser.add_argument('--verbose', action='store_true')
    # parser.add_argument('--num_frames', type=int, default=None, help='Path to dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')
    parser.add_argument('--base-path', '-p', type=str, default='/home/users/u6567085/data', help='base directory for dataset')
    parser.add_argument('--n-frames', '-f', type=int, default=256, help='number of frames sampled per video for train/val')
    parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')
    parser.add_argument('--activity', '-ac', type=str, nargs='+', required=True, help='activity classes to select for dataset')

    args = parser.parse_args()

    if os.path.isdir(args.model_path):
        ckpts = natsorted(glob.glob(os.path.join(args.model_path, '*')))
    else:
        ckpts = [args.model_path]
    
    
    ckpt_mul = args.device
    main(ckpts, args)