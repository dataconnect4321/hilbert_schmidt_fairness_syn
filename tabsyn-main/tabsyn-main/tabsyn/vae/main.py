import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from tabsyn.vae.flip_fairness import (
    disentanglement_loss,
    multi_stage_disentanglement_loss,
    sliced_wasserstein_distance,
    uniform_attribute_loss,
    uniform_attribute_loss_per_sample,
    BalancedGroupSampler,
    RDPAccountant,
    register_tabsyn_grad_samplers,
    get_dp_trainable_parameters,
)
from utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z,
                 s_idx=None, Recon_S_logits=None):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            # FLIP: exclude the protected attribute from the reconstruction
            # loss - its assignment should be random (Sec. 4.2), and its
            # logits are instead supervised by the uniform attribute loss.
            if s_idx is not None and idx == s_idx:
                continue
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())

    # FLIP uniform attribute loss L_S (Eq. 3): the per-sample variant is
    # used so that DP-SGD's per-sample clipping bounds each record's
    # influence (the batch-mean version couples all samples in the
    # batch, voiding the record-level guarantee). Minimizing the sum of
    # per-sample ||softmax - uniform||_2 pushes every individual
    # reconstruction toward uniform protected-attribute logits.
    loss_s = torch.tensor(0.0, device=X_num.device)
    if s_idx is not None and Recon_S_logits is not None:
        loss_s = uniform_attribute_loss_per_sample(Recon_S_logits)

    return mse_loss, ce_loss, loss_kld, acc, loss_s


def main(args):
    dataname = args.dataname
    data_dir = f'data/{dataname}'

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    device =  args.device


    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'])

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    # FLIP: identify the protected attribute column and per-sample group labels
    s_idx = args.sensitive_idx          # index into the categorical columns
    s_groups = None
    if s_idx is not None and s_idx >= 0:
        s_groups = X_train_cat[:, s_idx].long()
        print(f'Protected attribute: categorical column {s_idx} '
              f'with {len(torch.unique(s_groups))} groups')

    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    batch_size = 4096
    if s_groups is not None:
        # FLIP balanced mini-batch sampling (Sec. 4.2.1): equal group
        # representation per batch.
        sampler = BalancedGroupSampler(s_groups.numpy(), batch_size)
        train_loader = DataLoader(
            train_data,
            batch_sampler = sampler,
            num_workers = 4,
        )
        print(f'Balanced sampling: {sampler.num_batches} batches/epoch, '
              f'{sampler.per_group} per group x {sampler.n_groups} groups')
    else:
        train_loader = DataLoader(
            train_data,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4,
        )

    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True)
    model = model.to(device)

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    # FLIP: DP-SGD training under Renyi DP (Sec. 4.4 of arXiv:2508.21815).
    # When enabled, the optimizer is wrapped with Opacus's DPOptimizer:
    # per-sample gradients are L2-clipped to max_grad_norm, Gaussian
    # noise with std noise_multiplier * max_grad_norm is added, and the
    # privacy loss is composed over iterations via the RDP accountant.
    use_dp = args.dp and s_groups is not None
    accountant = None
    if use_dp:
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule
        # Native grad samplers for TabSyn's custom modules (Tokenizer,
        # Reconstructor), avoiding the functorch/vmap incompatibility
        # with nn.Embedding.
        register_tabsyn_grad_samplers()
        # Wrap the model so per-sample gradients are captured for clipping.
        # strict=False allows the Tokenizer's constant (non-trainable)
        # category offsets, which do not affect per-sample gradient
        # correctness.
        model = GradSampleModule(model, strict=False)
        # Exclude dead parameters (Transformer.head / last_normalization
        # are defined but never used in forward) - DPOptimizer rejects
        # params that never receive per-sample gradients.
        dp_params = get_dp_trainable_parameters(model)
        base_optimizer = torch.optim.Adam(dp_params, lr=LR, weight_decay=WD)
        optimizer = DPOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            expected_batch_size=batch_size,
        )
        # RDP accountant: sample rate = batch_size / n_train
        n_train = X_train_num.shape[0]
        accountant = RDPAccountant(
            noise_multiplier=args.noise_multiplier,
            sample_rate=batch_size / n_train,
            delta=args.dp_delta,
        )
        print(f'DP-SGD enabled: noise_multiplier={args.noise_multiplier}, '
              f'max_grad_norm={args.max_grad_norm}, delta={args.dp_delta}')
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    num_epochs = 4000
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    # FLIP two-phase training (Sec. 4.3):
    #   Phase 1 (quality): standard beta-VAE + uniform attribute loss L_S.
    #   Phase 2 (disentanglement): freeze a reference encoder q_theta0, then
    #   train with L_fair = SWD(q0(z|x), qt(z|x)) + lambda * (-CKA^T between
    #   groups), applied at the latent, detokenizer and decoder stages.
    phase1_epochs = args.phase1_epochs if s_groups is not None else num_epochs
    lambda_fair = args.lambda_fair
    reference_encoder = None  # frozen copy of the Phase-1 encoder

    beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        phase = 1 if epoch < phase1_epochs else 2
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Phase {phase}]")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0
        curr_loss_fair = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            # FLIP: multi-stage forward pass exposing the latent,
            # detokenizer and decoder representations for disentanglement.
            stages = model.forward_with_stages(batch_num, batch_cat)
            Recon_X_num = stages['recon_x_num']
            Recon_X_cat = stages['recon_x_cat']
            mu_z = stages['mu_z']
            std_z = stages['std_z']

            # Protected-attribute logits and group labels for this batch
            batch_s_logits = None
            batch_groups = None
            if s_idx is not None and s_idx >= 0:
                batch_s_logits = Recon_X_cat[s_idx]
                batch_groups = batch_cat[:, s_idx].long()

            loss_mse, loss_ce, loss_kld, train_acc, loss_s = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z,
                s_idx=s_idx, Recon_S_logits=batch_s_logits)

            if phase == 1:
                # Phase 1: quality (Eq. 9)
                loss = loss_mse + loss_ce + beta * loss_kld + loss_s
                loss_fair_val = torch.tensor(0.0)
            else:
                # Phase 2: disentanglement (Eq. 10)
                # Divergence penalty: SWD between frozen reference latents
                # and current latents.
                with torch.no_grad():
                    ref_mu, _ = reference_encoder(batch_num, batch_cat)
                ref_mu_flat = ref_mu.reshape(ref_mu.shape[0], -1)
                mu_flat = mu_z.reshape(mu_z.shape[0], -1)
                div_penalty = sliced_wasserstein_distance(ref_mu_flat, mu_flat)

                # Disentanglement: negative CKA^T between protected groups,
                # applied at the latent, detokenizer and decoder stages
                # (Sec. 4.3.2), mean-aggregated across stages.
                disent = multi_stage_disentanglement_loss(stages, batch_groups,
                                                           s_idx=s_idx)

                loss_fair_val = div_penalty + lambda_fair * disent
                loss = loss_mse + loss_ce + beta * loss_kld + loss_s + loss_fair_val

            loss.backward()
            optimizer.step()

            # FLIP: record one DP-SGD iteration for the RDP accountant
            if accountant is not None:
                accountant.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length
            curr_loss_fair  += float(loss_fair_val) * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count
        fair_loss = curr_loss_fair / curr_count

        # Freeze the reference encoder at the Phase-1 -> Phase-2 boundary
        if phase == 1 and epoch + 1 == phase1_epochs:
            reference_encoder = Encoder_model(
                NUM_LAYERS, d_numerical, categories, D_TOKEN,
                n_head=N_HEAD, factor=FACTOR).to(device)
            reference_encoder.load_weights(model)
            for p in reference_encoder.parameters():
                p.requires_grad_(False)
            reference_encoder.eval()
            print(f'--- Phase 1 complete: reference encoder frozen at epoch {epoch+1} ---')
        

        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            val_s_logits = Recon_X_cat[s_idx] if (s_idx is not None and s_idx >= 0) else None
            val_mse_loss, val_ce_loss, val_kl_loss, val_acc, val_s_loss = compute_loss(
                X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z,
                s_idx=s_idx, Recon_S_logits=val_s_logits)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd


        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        privacy_msg = ''
        if accountant is not None:
            eps_spent, alpha_opt = accountant.get_privacy_spent()
            privacy_msg = ', eps={:.4f} (alpha={})'.format(eps_spent, alpha_opt)
        print('epoch: {}, phase: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Fair:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}{}'.format(epoch, phase, beta, num_loss, cat_loss, kl_loss, fair_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item(), privacy_msg ))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    if accountant is not None:
        eps_spent, alpha_opt = accountant.get_privacy_spent()
        print(f'Final privacy guarantee: (epsilon={eps_spent:.4f}, delta={args.dp_delta}) '
              f'at Renyi order alpha={alpha_opt}')
    
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print('Successfully load and save the model!')

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    # FLIP fairness arguments (arXiv:2508.21815)
    parser.add_argument('--sensitive_idx', type=int, default=-1,
                        help='Index of the protected attribute among the categorical '
                             'columns. -1 disables all fairness interventions.')
    parser.add_argument('--phase1_epochs', type=int, default=2000,
                        help='Number of quality-focused (Phase 1) epochs before the '
                             'disentanglement (Phase 2) stage begins.')
    parser.add_argument('--lambda_fair', type=float, default=1.0,
                        help='Weight of the CKA^T disentanglement term in Phase 2.')

    # FLIP privacy arguments (DP-SGD under Renyi DP, Sec. 4.4)
    parser.add_argument('--dp', action='store_true',
                        help='Enable DP-SGD training with Renyi DP accounting.')
    parser.add_argument('--noise_multiplier', type=float, default=1.0,
                        help='DP-SGD Gaussian noise multiplier (sigma).')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='DP-SGD per-sample gradient L2 clipping norm.')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='Target delta for the (epsilon, delta)-DP guarantee.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'