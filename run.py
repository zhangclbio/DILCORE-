import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import os
import csv
import warnings
from datetime import datetime
import utils2
import utils
import load_data
from modelInfoNCE5 import DILCR, loss_funcation

warnings.filterwarnings("ignore")
DATASET_PATH = "/home/"
# seed = np.random.randint(0, 10000)  # 生成不同的随机种子
seed = 123456
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    cancer_type = "kirc"
    conf = dict()
    conf['dataset'] = cancer_type
    exp, methy, mirna, survival = load_data.load_TCGA(DATASET_PATH+"data/", cancer_type,'mean') # Preprocessing method  
    # 输出样本数量和基因数量
    sample_num = exp.shape[1]
    gene_num = exp.shape[0]
    print(f"样本数量1: {sample_num}")
    print(f"基因数量: {gene_num}")
    methy_sample_num = methy.shape[1]
    methy_num = methy.shape[0]
    print(f"样本数量2: {methy_sample_num}")
    print(f"甲基数量: {methy_num}")
    mirna_sample_num = exp.shape[1]
    mirna_num = exp.shape[0]
    print(f"样本数量3: {mirna_sample_num}")
    print(f"mirna数量: {mirna_num}")     
    exp_df = torch.tensor(exp.values.T, dtype=torch.float32).to(device)
    methy_df = torch.tensor(methy.values.T, dtype=torch.float32).to(device)
    mirna_df = torch.tensor(mirna.values.T, dtype=torch.float32).to(device)
    full_data = [utils2.p_normalize(exp_df), utils2.p_normalize(methy_df), utils2.p_normalize(mirna_df)]
    
    # params
    conf = dict()
    conf['dataset'] = cancer_type
    # conf['β']=2
    conf['view_num'] = 3
    conf['batch_size'] = 128
    conf['encoder_dim'] = [1024]
    conf['feature_dim'] = 512
    conf['peculiar_dim'] = 128
    conf['common_dim'] = 128
    conf['mu_logvar_dim'] = 10
    conf['cluster_var_dim'] = 3 * conf['common_dim']
    conf['up_and_down_dim'] = 512
    conf['use_cuda'] = True
    conf['stop'] = 1e-6
    eval_epoch = 500
    lmda_list = dict()
    lmda_list['rec_lmda'] = 0.9
    lmda_list['KLD_lmda'] = 0.3
    lmda_list['I_loss_lmda'] = 0.1
    conf['kl_loss_lmda'] = 10
    conf['update_interval'] = 50
    conf['lr'] = 1e-4
    conf['pre_epochs'] = 1500
    conf['idec_epochs'] = 500
    # If the DILCR effect is not good, we recommend adjusting the preprocessing epoch.
    if conf['dataset'] == "aml":
        conf['cluster_num'] = 3
    if conf['dataset'] == "brca":
        conf['cluster_num'] = 5
    if conf['dataset'] == "skcm":
        conf['cluster_num'] = 5
    if conf['dataset'] == "lihc":
        conf['cluster_num'] = 5
    if conf['dataset'] == "coad":
        conf['cluster_num'] = 4
    if conf['dataset'] == "kirc":
        conf['cluster_num'] = 4
    if conf['dataset'] == "gbm":
        conf['cluster_num'] = 3
    if conf['dataset'] == "ov":
        conf['cluster_num'] = 3
    if conf['dataset'] == "lusc":
        conf['cluster_num'] = 3
    if conf['dataset'] == "sarc":
        conf['cluster_num'] = 5
    # seed = np.random.randint(0, 10000)  # 生成不同的随机种子
    setup_seed(seed=seed)
    
    # ========================Result File====================
    folder = "/home//result/{}_result".format(conf['dataset'])
    if not os.path.exists(folder):
        os.makedirs(folder)
    # current_time = datetime.now().strftime("%m%d_%H%M")
    # result = open("{}/{}_{}_{}.csv".format(folder, conf['dataset'], conf['cluster_num'], current_time), 'w+')
    result = open("{}/{}_{}.csv".format(folder, conf['dataset'], conf['cluster_num']), 'w+')
    writer = csv.writer(result)
    writer.writerow(['p', 'logp', 'log10p', 'epoch', 'step'])
    # =======================Initialize the model and loss function====================
    in_dim = [exp_df.shape[1], methy_df.shape[1], mirna_df.shape[1]]
    model = DILCR(in_dim=in_dim, encoder_dim=conf['encoder_dim'], feature_dim=conf['feature_dim'],
                  common_dim=conf['common_dim'],
                  mu_logvar_dim=conf['mu_logvar_dim'], cluster_var_dim=conf['cluster_var_dim'],
                  up_and_down_dim=conf['up_and_down_dim'], cluster_num=conf['cluster_num'],
                  peculiar_dim=conf['peculiar_dim'], view_num=conf['view_num'], device = device)
    model = model.to(device=device)
    opt = torch.optim.AdamW(lr=conf['lr'], params=model.parameters())
    loss = loss_funcation()
    # =======================pre-training VAE====================
    print("pre-----------------------train-dataset-: {} cluster_num-: {}".format(conf['dataset'], conf['cluster_num']))
    pbar = tqdm(range(conf['pre_epochs']), ncols=120)
    max_log = 0.0
    max_label = []
    for epoch in pbar:
        # 抽取数据 训练batch
        sample_num = exp_df.shape[0]
        randidx = torch.randperm(sample_num)
        for i in range(round(sample_num / conf['batch_size'])):
            idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
            data_batch = [utils2.p_normalize(exp_df[idx]), utils2.p_normalize(methy_df[idx]), utils2.p_normalize(mirna_df[idx])]
            # 前向传播
            out_list, latent_dist = model(data_batch)

            # l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
            #                     latent_dist=latent_dist,
            #                     lmda_list=lmda_list, batch_size=conf['batch_size'])
            l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
                    latent_dist=latent_dist, lmda_list=lmda_list, batch_size=conf['batch_size'], model=model)
            # 反向传播
            l.backward()
            # 添加梯度监控（每个batch执行）
            # if (epoch % 500 == 0) and (i == 0):  # 每50个epoch监控第一个batch
            #     print(f"\nEpoch {epoch} Gradients:")
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             print(f"{name}: {param.grad.abs().mean().item():.2e}")
            opt.step()
            opt.zero_grad()
        # Evaluation model
        # 每500轮评估聚类效果
        if (epoch + 1) % eval_epoch == 0:
            with torch.no_grad():
                model.eval()
                out_list, latent_dist = model(full_data)
                kmeans = KMeans(n_clusters=conf['cluster_num'], n_init=20, random_state=seed, init="k-means++")
                kmeans.fit(latent_dist['cluster_var'].cpu().numpy())
                pred = kmeans.labels_
                cluster_center = kmeans.cluster_centers_
                survival["label"] = np.array(pred)
                df = survival
                res = utils2.log_rank(df)
                writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "pre"])
                result.flush()
                model.train()
            # 保存最佳模型
            if (res['log10p'] > max_log):
                max_log = res['log10p']
                max_label = pred
                torch.save(model.state_dict(), "{}/{}_max_log.pdparams".format(folder, conf['dataset']))

        pbar.set_postfix(loss="{:3.4f}".format(loss_dict['loss'].item()),
                         rec_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
                         KLD="{:3.4f}".format(loss_dict['KLD'].item()),
                         I_loss="{:3.4f}".format(loss_dict['I_loss'].item()))
    # =======================training IDEC=====================
    # ========================Initialize cluster centers via K-means==============
    #聚类优化阶段（IDEC）
    out_list, latent_dist = model(full_data)
    print("idec-----------------------train-dataset-: {} cluster_num-: {}".format(conf['dataset'], conf['cluster_num']))
    # 用预训练特征初始化聚类中心
    kmeans = KMeans(n_clusters=conf['cluster_num'], random_state=seed, init="k-means++").fit(
        latent_dist['cluster_var'].detach().cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    y_pred_last = kmeans.labels_
    max_label_log = 0.0
    max_label_pred = y_pred_last

    pbar = tqdm(range(conf['idec_epochs']), ncols=120)
    for epoch in pbar:# 每50轮更新软分配
        if epoch % conf['update_interval'] == 0:
            with torch.no_grad():
                _, latent_dist = model(full_data)
                # 计算软分配概率q
                tmp_q = latent_dist['q']
                y_pred = tmp_q.cpu().numpy().argmax(1)
                weight = tmp_q ** 2 / tmp_q.sum(0)
                # 计算目标分布p
                p = (weight.t() / weight.sum(1)).t()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                # 评估
                df = survival
                df["label"] = np.array(y_pred)
                res = utils2.log_rank(df)
                writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "IDEC"])
                result.flush()
                if res['log10p'] > max_label_log:
                    print(f"log10p: {res['log10p']}")
                    max_label_log = res['log10p']
                    # print(max_label_log)
                    max_label_pred = y_pred
                    torch.save(model.state_dict(), "{}/{}_max_label_log.pdparams".format(folder, conf['dataset']))

                if epoch > 0 and delta_label < conf['stop']:
                    print('delta_label {:.4f}'.format(delta_label), '< tol',
                        conf['stop'])
                    print('Reached tolerance threshold. Stopping training.')
                    break
                save_dir = DATASET_PATH+"result/"
                utils2.visualize_latent_space(latent_dist, epoch, save_dir=DATASET_PATH+"result/{}_result/latent_space_images.png".format(conf['dataset']), title="Latent Space", method='tSNE',
                                              labels=pred)
                # utils2.lifeline_analysis(df, title_g=conf['dataset']+" Survival Analysis", save_path=DATASET_PATH+"result/survival_analysis.png")            
                utils2.lifeline_analysis(df, title_g=conf['dataset']+" Survival Analysis", save_path=DATASET_PATH+"result/{}_result/survival_analysis.png".format(conf['dataset']))            

        # 抽取数据 训练batch 小批量训练（加入KL散度）
        sample_num = exp_df.shape[0]
        randidx = torch.randperm(sample_num)
        for i in range(round(sample_num / conf['batch_size'])):
            idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
            data_batch = [utils2.p_normalize(exp_df[idx]), utils2.p_normalize(methy_df[idx]), utils2.p_normalize(mirna_df[idx])]
            out_list, latent_dist = model(data_batch)
            kl_loss = F.kl_div(latent_dist['q'].log(), p[idx])
            # l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
            #                     latent_dist=latent_dist,
            #                     lmda_list=lmda_list, batch_size=conf['batch_size'])
            l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
                    latent_dist=latent_dist, lmda_list=lmda_list, batch_size=conf['batch_size'], model=model)
            l = conf['kl_loss_lmda'] * kl_loss
            l.backward()
            # 添加梯度监控（每个update interval执行）
            # if (epoch % 100 == 0) and (i == 0):
            #     print(f"\nIDEC Epoch {epoch} Gradients:")
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             print(f"{name}: {param.grad.abs().mean().item():.2e}")
            opt.step()
            opt.zero_grad()

        # scheduler.step()
        pbar.set_postfix(loss="{:3.4f}".format(loss_dict['loss'].item()),
                         rec_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
                         KLD="{:3.4f}".format(loss_dict['KLD'].item()),
                         I_loss="{:3.4f}".format(loss_dict['I_loss'].item()),
                         KL_loss="{:3.4f}".format(kl_loss.item()))
    survival["label"] = np.array(max_label)
    clinical_data = utils2.get_clinical(DATASET_PATH + "data/clinical", survival, conf["dataset"])
    cnt_NI = utils2.clinical_enrichement(clinical_data['label'],clinical_data)
    survival["label"] = np.array(max_label_pred)
    clinical_data = utils2.get_clinical(DATASET_PATH + "data/clinical", survival, conf["dataset"])
    # clinical_data["pathologic_T"] = clinical_data["pathologic_T"].replace({
    #     "T1a": "T1", "T1b": "T1",
    #     "T2a": "T2", "T2b": "T2",
    #     "T3a": "T3", "T3b": "T3",
    #     "T4a": "T4", "T4b": "T4"
    # })
    # 去除 MX 和 NX 无效数据
    # clinical_data = clinical_data[clinical_data["pathologic_M"] != "MX"]
    # clinical_data = clinical_data[clinical_data["pathologic_N"] != "NX"]
    clinical_data.to_csv("clinical_data.csv", index=False)
    cnt = utils2.clinical_enrichement(clinical_data['label'],clinical_data)
    writer.writerow([res['p'], res['log2p'], res['log10p'], epoch, "IDEC"])
    print("{}:    DILCR-NI:  {}/{:.1f}   DILCR-ALL:   {}/{:.1f}".format(conf['dataset'],cnt_NI,max_log,cnt,max_label_log))
    #DILCR-NI可能代表仅使用预训练阶段（无监督，无标签指导）的聚类结果，而DILCR-ALL是结合了IDEC阶段（有标签分布优化）的最终结果,NI可能指“No Improvement”或某种基线，而ALL表示完整流程。
    #基于预训练阶段的纯无监督聚类结果;经过IDEC阶段聚类优化的最终结果
    # 获取样本 ID，假设 exp 的索引是样本 ID
    exp = exp.T
    sample_ids = exp.index   
    # 保存 IDEC 训练阶段的聚类标签
    idec_clustering_result = pd.DataFrame({
        'Sample_ID': sample_ids,
        'IDEC_Cluster_Label': max_label_pred
    })
    idec_clustering_result.to_csv(f"{folder}/{conf['dataset']}_idec_clustering_result.csv", index=False)

    # 关闭之前打开的结果文件
    result.close()