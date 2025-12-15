CONFIG = {
    'kirc': {
        'cluster_num': 4,
        'λ_rec': 0.9,
        'λ_KLD': 0.3,
        'λ_I_loss': 10,
        'pre_epochs': 2000,
        'idec_epochs': 1000,
        'lr': 1e-4,
        'batch_size': 128,
        'update_interval': 50,
        'beta': 1.5,
        'cnt_threshold': 1.0,         # 针对kirc的cnt阈值
        'max_log_threshold': 10.0      # 针对kirc的max_log阈值
    },
    'brca': {
        'cluster_num': 5,
        'λ_rec': 0.8,
        'λ_KLD': 0.2,
        'λ_I_loss': 10,
        'pre_epochs': 2000,
        'idec_epochs': 1500,
        'lr': 1e-4,
        'batch_size': 128,
        'update_interval': 10,
        'beta': 1.0,
        'cnt_threshold': 4.0,
        'max_log_threshold': 5.56
    },
    'skcm': {
        'cluster_num': 5,
        'λ_rec': 0.8,
        'λ_KLD': 0.2,
        'λ_I_loss': 0.4,
        'pre_epochs': 2000,
        'idec_epochs': 1500,
        'lr': 1e-4,
        'batch_size': 256,
        'update_interval': 10,
        'beta': 1.5,
        'cnt_threshold': 4.0,
        'max_log_threshold': 10.78
    },
    'lihc': {
        'cluster_num': 5,
        'λ_rec': 0.7,
        'λ_KLD': 0.3,
        'λ_I_loss': 0.3,
        'pre_epochs': 1000,
        'idec_epochs': 500,
        'lr': 1e-4,
        'batch_size': 128,
        'update_interval': 50,
        'beta': 1.0,
        'cnt_threshold': 3.0,
        'max_log_threshold': 4.77
    },
    'coad': {
        'cluster_num': 4,
        'λ_rec': 0.9,
        'λ_KLD': 0.2,
        'λ_I_loss': 0.3,
        'pre_epochs': 2000,
        'idec_epochs': 1500,
        'lr': 1e-4,
        'batch_size': 512,
        'update_interval': 50,
        'beta': 1.0,
        'cnt_threshold': 2.0,
        'max_log_threshold': 1.99
    },
    'gbm': {
        'cluster_num': 3,
        'λ_rec': 0.7,
        'λ_KLD': 0.3,
        'λ_I_loss': 0.4,
        'pre_epochs': 2500,
        'idec_epochs': 1500,
        'lr': 1e-4,
        'batch_size': 512,
        'update_interval': 10,
        'beta': 1.0,
        'cnt_threshold': 1.0,
        'max_log_threshold': 4.89
    },
    'ov': {
        'cluster_num': 3,
        'λ_rec': 0.9,
        'λ_KLD': 0.2,
        'λ_I_loss': 10,
        'pre_epochs': 1500,
        'idec_epochs': 1000,
        'lr': 1e-4,
        'batch_size': 256,
        'update_interval': 50,
        'beta': 1.0,
        'cnt_threshold': 1.0,
        'max_log_threshold': 2.11
    },
    'lusc': {
        'cluster_num': 3,
        'λ_rec': 0.9,
        'λ_KLD': 0.3,
        'λ_I_loss': 0.1,
        'pre_epochs': 1000,
        'idec_epochs': 500,
        'lr': 1e-4,
        'batch_size': 512,
        'update_interval': 50,
        'beta': 1.0,
        'cnt_threshold': 3.0,
        'max_log_threshold': 1.67
    },
    'aml': {
        'cluster_num': 3,
        'λ_rec': 0.9,
        'λ_KLD': 0.3,
        'λ_I_loss': 10,
        'pre_epochs': 2000,
        'idec_epochs': 1000,
        'lr': 1e-4,
        'batch_size': 256,
        'update_interval': 10,
        'beta': 1.0,
        'cnt_threshold': 1.0,
        'max_log_threshold': 9.74
    },
    'sarc': {
        'cluster_num': 5,
        'λ_rec': 0.9,
        'λ_KLD': 0.3,
        'λ_I_loss': 10,
        'pre_epochs': 1000,
        'idec_epochs': 500,
        'lr': 1e-4,
        'batch_size': 512,
        'update_interval': 50,
        'beta': 1.0,
        'cnt_threshold': 1.0,
        'max_log_threshold': 3.08
    }
}

# 全局默认设置
GLOBAL_SETTINGS = {
    'default_seed': 123456,
    'device': 'cuda:1',
    'default_cnt_threshold': 0.6,       # 如果数据集配置中没有指定阈值，使用此默认值
    'default_max_log_threshold': 2.5    # 如果数据集配置中没有指定阈值，使用此默认值
}