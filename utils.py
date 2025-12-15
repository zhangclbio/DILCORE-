import difflib
import math
import os
import re
import torch
from scipy.stats import kruskal, chi2_contingency, fisher_exact
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, chi2_contingency
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pair_confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from umap import UMAP

def p_normalize(x, p=2):
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)

# def lifeline_analysis(df, title_g="brca", save_path="./result/survival_analysis.png"):
#     '''
#     生存分析画图并保存
#     :param df: 生存分析数据，DataFrame 格式，包含 label、Survival、Death 字段
#     :param title_g: 图标题
#     :param save_path: 保存图像的路径（包括文件名）
#     '''
#     n_groups = len(set(df["label"]))
#     kmf = KaplanMeierFitter()
#     plt.figure()  # 创建新的图像
#     for group in range(n_groups):
#         idx = (df["label"] == group)
#         kmf.fit(df['Survival'][idx], df['Death'][idx], label='class_' + str(group))
#         kmf.plot()

#     plt.title(title_g)
#     plt.xlabel("lifeline (days)")
#     plt.ylabel("survival probability")
#     plt.tight_layout()  # 自动调整布局

#     # 保存图片到指定路径
#     plt.savefig(save_path)
#     plt.close()  # 关闭当前图像，释放资源
#     print(f"Survival analysis plot saved to {save_path}")

def lifeline_analysis(df, title_g="BRCA", p_value=None, save_path="./result/survival_analysis.pdf"):
    '''
    生存分析画图并保存为 PDF，并在图中添加 p 值信息
    :param df: DataFrame，包含 'label'（类别）、'Survival'（生存时间）、'Death'（是否死亡）
    :param title_g: 图标题
    :param p_value: 浮点数，log-rank 检验的 p 值
    :param save_path: 图像保存路径（建议使用 .pdf 格式）
    '''
    sns.set(style="whitegrid")
    n_groups = len(set(df["label"]))
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=n_groups)

    for group in range(n_groups):
        idx = (df["label"] == group)
        kmf.fit(df['Survival'][idx], df['Death'][idx], label=f'Subtype {group}')  # 修改此处的Class为Subtype
        kmf.plot(ci_show=False, lw=2, color=palette[group])

    plt.title(f"{title_g}", fontsize=16)
    plt.xlabel("Time (days)", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Group", fontsize=12, title_fontsize=12, loc='best')

    # 显示 p 值
    if p_value is not None:
        if p_value < 0.0001:
            p_text = "p < 0.0001"
        else:
            p_text = f"p = {p_value:.4f}"
        plt.text(0.05, 0.05, p_text, transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Survival analysis plot saved to {save_path}")

# 富集分析
def clinical_enrichementraw(label,clinical):
    cnt = 0
    # age 连续 使用KW检验
    print(label,clinical)
    stat, p_value_age = kruskal(np.array(clinical["age"]), np.array(label))
    if p_value_age < 0.05:
        cnt += 1
        print("---age---")
    # 其余离散 卡方检验
    stat_names = ["gender","pathologic_T","pathologic_M","pathologic_N","pathologic_stage"]
    for stat_name in stat_names:
        if stat_name in clinical:
            c_table = pd.crosstab(clinical[stat_name],label,margins = True)
            stat, p_value_other, dof, expected = chi2_contingency(c_table)
            if p_value_other < 0.05:
                cnt += 1
                print(f"---{stat_name}---")
    return cnt

def clinical_enrichement(label, clinical):
    cnt = 0
    
    # -------------------------
    # 1. 连续变量：年龄（KW检验）
    # -------------------------
    valid_age_idx = clinical["age"].notna()  # 过滤年龄缺失值
    if valid_age_idx.sum() >= 2:  # KW检验需至少2组，每组≥1样本
        age_data = clinical.loc[valid_age_idx, "age"].values
        label_age = label[valid_age_idx].values
        stat, p_value_age = kruskal(age_data, label_age)
        if p_value_age < 0.05:
            cnt += 1
            print("---age (Kruskal-Wallis)---")
    
    # -------------------------
    # 2. 离散变量：性别、分期等
    # -------------------------
    stat_names = ["gender", "pathologic_T", "pathologic_M", "pathologic_N", "pathologic_stage"]
    for stat_name in stat_names:
        if stat_name not in clinical.columns:
            continue  # 列不存在，跳过
        
        # 过滤当前变量的缺失值，同步label
        valid_idx = clinical[stat_name].notna()
        if valid_idx.sum() == 0:
            continue  # 全部缺失，跳过
        
        var_data = clinical.loc[valid_idx, stat_name]
        label_sub = label[valid_idx]
        
        # -------------------------
        # 特殊处理：性别（二分类）
        # -------------------------
        if stat_name == "gender":
            unique_gender = var_data.unique()
            # 情况1：性别为二分类（如MALE/FEMALE）
            if len(unique_gender) == 2:
                c_table = pd.crosstab(var_data, label_sub)
                # 仅当列联表是2×2时，用Fisher检验
                if c_table.shape == (2, 2):  
                    odds_ratio, p_value = fisher_exact(c_table)
                    if p_value < 0.05:
                        cnt += 1
                        print(f"---{stat_name} (Fisher's Exact Test)---")
                # 情况2：性别类别数≠2（罕见，如含其他标识），回落卡方
                else:  
                    stat, p_value, dof, expected = chi2_contingency(c_table)
                    if p_value < 0.05:
                        cnt += 1
                        print(f"---{stat_name} (Chi-square)---")
        
        # -------------------------
        # 其他离散变量（直接卡方检验）
        # -------------------------
        else:
            c_table = pd.crosstab(var_data, label_sub)
            # 卡方检验（自动处理多分类）
            stat, p_value, dof, expected = chi2_contingency(c_table)
            if p_value < 0.05:
                cnt += 1
                print(f"---{stat_name} (Chi-square)---")
    
    return cnt

def log_rank(df):
    '''
    :param df: 传入生存数据
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :return: res 包含了p log2p log10p
    '''
    res = dict()
    results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
    res['p'] = results.summary['p'].item()
    res['log10p'] = -math.log10(results.summary['p'].item())
    res['log2p'] = -math.log2(results.summary['p'].item())
    return res

# def get_clinical(path,survival,cancer_type):
#     clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
#     if cancer_type == 'kirc':
#         replace = {'gender.demographic': 'gender','submitter_id.samples': 'sampleID'}
#         clinical = clinical.rename(columns=replace)  # 为某个 index 单独修改名称
#         clinical["sampleID"] = [re.sub("A", "", x) for x in clinical["sampleID"].str.upper()]
#     clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]
#     survival['age'] = pd.NA # 初始化年龄
#     survival['gender'] = pd.NA # 初始化年龄
#     if 'pathologic_T' in clinical.columns:
#         survival['T'] = pd.NA # 初始化年龄
#     if 'pathologic_M' in clinical.columns:
#         survival['M'] = pd.NA # 初始化年龄
#     if 'pathologic_N' in clinical.columns:
#         survival['N'] = pd.NA # 初始化年龄
#     if 'tumor_stage.diagnoses' in clinical.columns:
#         survival['stage'] = pd.NA # 初始化年龄
#     i = 0
#     # 找对应的参数
#     for name in survival['PatientID']:
#         # print(name)
#         flag = difflib.get_close_matches(name,list(clinical["sampleID"]),1,cutoff=0.6)
#         if flag:
#             idx = list(clinical["sampleID"]).index(flag[0])
#             survival['age'][i] = clinical['age_at_initial_pathologic_diagnosis'][idx]
#             survival['gender'][i] = clinical['gender'][idx]
#             if 'pathologic_T' in clinical.columns:
#                 survival['T'][i] = clinical['pathologic_T'][idx]
#             if 'pathologic_M' in clinical.columns:
#                 survival['M'][i] = clinical['pathologic_M'][idx]
#             if 'pathologic_N' in clinical.columns:
#                 survival['N'][i] = clinical['pathologic_N'][idx]
#             if 'tumor_stage.diagnoses' in clinical.columns:
#                 survival['stage'][i] = clinical['tumor_stage.diagnoses'][idx]
#         else: print(name)
#         i = i + 1
#     return survival.dropna(axis=0, how='any')

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
    if cancer_type == 'kirc':
        replace = {'gender.demographic': 'gender','submitter_id.samples': 'sampleID'}
        clinical = clinical.rename(columns=replace)  # 为某个 index 单独修改名称
        clinical["sampleID"] = [re.sub("A", "", x) for x in clinical["sampleID"].str.upper()]
    clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]
    survival['age'] = pd.NA # 初始化年龄
    survival['gender'] = pd.NA # 初始化年龄
    if 'pathologic_T' in clinical.columns:
        survival['pathologic_T'] = pd.NA # 初始化年龄
    if 'pathologic_M' in clinical.columns:
        survival['pathologic_M'] = pd.NA # 初始化年龄
    if 'pathologic_N' in clinical.columns:
        survival['pathologic_N'] = pd.NA # 初始化年龄
    if 'tumor_stage.diagnoses' in clinical.columns:
        survival['pathologic_stage'] = pd.NA # 初始化年龄
    # 分期处理函数
    def process_stage(value):
        if pd.isna(value) or value.strip() in ['None', '']:
            return pd.NA
        # 处理 M 分期（如 cM0 (i+) → M0）
        if 'M' in value:
            return re.sub(r'[^M01]', '', value) or pd.NA
        # 处理 T 和 N 分期（如 T1a → T1, N2b → N2）
        elif any(c in value for c in ['T', 'N']):
            return re.sub(r'[a-d\(\)\s\+]', '', value) or pd.NA
        return value
    
    # 处理临床数据中的分期列
    if 'pathologic_T' in clinical.columns:
        clinical['pathologic_T'] = clinical['pathologic_T'].apply(process_stage)
    if 'pathologic_M' in clinical.columns:
        clinical['pathologic_M'] = clinical['pathologic_M'].apply(process_stage)
    if 'pathologic_N' in clinical.columns:
        clinical['pathologic_N'] = clinical['pathologic_N'].apply(process_stage)
    i = 0
    # 找对应的参数
    for name in survival['PatientID']:
        # print(name)
        flag = difflib.get_close_matches(name,list(clinical["sampleID"]),1,cutoff=0.6)
        if flag:
            idx = list(clinical["sampleID"]).index(flag[0])
            survival['age'][i] = clinical['age_at_initial_pathologic_diagnosis'][idx]
            survival['gender'][i] = clinical['gender'][idx]
            if 'pathologic_T' in clinical.columns:
                survival['pathologic_T'][i] = clinical['pathologic_T'][idx]
            if 'pathologic_M' in clinical.columns:
                survival['pathologic_M'][i] = clinical['pathologic_M'][idx]
            if 'pathologic_N' in clinical.columns:
                survival['pathologic_N'][i] = clinical['pathologic_N'][idx]
            if 'tumor_stage.diagnoses' in clinical.columns:
                survival['pathologic_stage'][i] = clinical['tumor_stage.diagnoses'][idx]
        else: print(name)
        i = i + 1
    return survival.dropna(axis=0, how='all')

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
     (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
     ri = (tp + tn) / (tp + tn + fp + fn)
     ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
     p, r = tp / (tp + fp), tp / (tp + fn)
     f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
     return ri, ari, f_beta

def cluster_evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_measure = get_rand_index_and_f_measure(label,pred)[2]
    return nmi, ari, acc, pur,f_measure
def visualize_latent_space(latent_dist, epoch, save_dir="latent_space_images", 
                          title="Latent Space Visualization", method='tSNE',
                          labels=None, umap_kwargs=None):
    """
    使用 t-SNE 或 PCA 可视化潜在空间，并将图像保存到文件
    Args:
        latent_dist: 模型输出的潜在空间分布，例如 latent_dist['cluster_var']
        epoch: 当前训练周期，用于标记图像
        save_dir: 保存图像的目录
        title: 图像标题
        method: 使用的降维方法 ('tSNE' 或 'PCA')
        labels: 用于颜色编码的标签（例如聚类标签）
    """
    seed = 123456
    latent_var = latent_dist['cluster_var'].cpu().detach().numpy()  # 获取潜在变量
    # 参数预处理（新增）
    umap_defaults = {'n_neighbors':15, 'min_dist':0.1, 'random_state':seed}
    umap_kwargs = umap_kwargs or {}
    umap_params = {**umap_defaults, **umap_kwargs}
    if method == 'tSNE':
        tsne = TSNE(n_components=2, random_state=seed)
        latent_2d = tsne.fit_transform(latent_var)
    elif method == 'PCA':
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_var)
    elif method == 'UMAP':  # 新增分支
        reducer = UMAP(n_components=2, **umap_params)
        latent_2d = reducer.fit_transform(latent_var)
    else:
        raise ValueError("Supported methods: 'tSNE', 'PCA', 'UMAP'")

    plt.figure(figsize=(8, 6))
    # 如果 labels 不为空，使用 labels 作为颜色编码
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', s=20)
    plt.title(f"{title} - Epoch {epoch}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.colorbar(label="Cluster Label")
    plt.grid(True)

    # 保存图像
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"latent_space_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()  # 关闭当前图像，避免内存溢出