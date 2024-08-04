import torch

params = [ 'attn.proj_query.weight', 'attn.proj_out.weight', 'attn.proj_val.weight', 'attn.proj_key.weight', 'smoe.gate.gate.weight', 'smoe.gate.router_layer.weight', 'smoe.gate.rnn_projector.weight', 'smoe.gate.gate.bias', 'smoe.experts.htoh4.weight', 'smoe.experts.htoh4.bias', 'smoe.experts.h4toh.weight', 'smoe.experts.h4toh.bias', 'smoe.layer_norm.weight', 'smoe.layer_norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'norm3.weight', 'norm3.bias']

n_layer = 24

RMoE = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-s-rnn-{n_layer}/smoe-rnn.pt')['model']
RMoE_np = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-s-rnn-{n_layer}-np/smoe-rnn.pt')['model']
RMoE_np_r05 = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-s-rnn-{n_layer}-np-r0.5/smoe-rnn.pt')['model']
SMoE = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-{n_layer}/smoe.pt')['model']

statistic_RMoE = {}
for name, param in RMoE.items():
    # if param.requires_grad:
        statistic_RMoE[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }
        
statistic_RMoE_np = {}
for name, param in RMoE_np.items():
    # if param.requires_grad:
        statistic_RMoE_np[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }

statistic_RMoE_np_r05 = {}
for name, param in RMoE_np_r05.items():
    # if param.requires_grad:
        statistic_RMoE_np_r05[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }

statistic_SMoE = {}
for name, param in SMoE.items():
    # if param.requires_grad:
        statistic_SMoE[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }
        
torch.save({'SMoE': statistic_SMoE, 'RMoE': statistic_RMoE, 'RMoE_np': statistic_RMoE_np, 'RMoE_np_r05': statistic_RMoE_np_r05}, f'ana_norm_{n_layer}.pt')


n_layer = 18

RMoE = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-s-rnn-{n_layer}/smoe-rnn.pt')['model']
RMoE_np = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-s-rnn-{n_layer}-np/smoe-rnn.pt')['model']
RMoE_np_r05 = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-s-rnn-{n_layer}-np-r0.5/smoe-rnn.pt')['model']
SMoE = torch.load(f'/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/zyz/CompeteSMoE/checkpoints-up/enwik8/transformers-s/smoe-{n_layer}/smoe.pt')['model']

statistic_RMoE = {}
for name, param in RMoE.items():
    # if param.requires_grad:
        statistic_RMoE[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }
        
statistic_RMoE_np = {}
for name, param in RMoE_np.items():
    # if param.requires_grad:
        statistic_RMoE_np[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }

statistic_RMoE_np_r05 = {}
for name, param in RMoE_np_r05.items():
    # if param.requires_grad:
        statistic_RMoE_np_r05[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }

statistic_SMoE = {}
for name, param in SMoE.items():
    # if param.requires_grad:
        statistic_SMoE[name] = {
            'norm': param.norm().item(),
            'std': param.std().item(),
            # 'max': param.max().item(),
            # 'min': param.min().item()
        }
        
torch.save({'SMoE': statistic_SMoE, 'RMoE': statistic_RMoE, 'RMoE_np': statistic_RMoE_np, 'RMoE_np_r05': statistic_RMoE_np_r05}, f'ana_norm_{n_layer}.pt')



import os
SMoE_name = f'./compare_norm/{n_layer}'
import matplotlib.pyplot as plt
if not os.path.exists(SMoE_name):
    os.makedirs(SMoE_name)
for param in params:
    input_param = [f'{i}.{param}' for i in range(0, n_layer)]
    
    statistic_input_param_SMoE = {k: v for k, v in statistic_SMoE.items() if any([name in k for name in input_param])}

    norm_SMoE = [v['norm'] for v in statistic_input_param_SMoE.values()]
    mean_SMoE = [v['mean'] for v in statistic_input_param_SMoE.values()]
    std_SMoE = [v['std'] for v in statistic_input_param_SMoE.values()]


    statistic_input_param_RMoE = {k: v for k, v in statistic_RMoE.items() if any([name in k for name in input_param])}

    norm_RMoE = [v['norm'] for v in statistic_input_param_RMoE.values()]
    mean_RMoE = [v['mean'] for v in statistic_input_param_RMoE.values()]
    std_RMoE = [v['std'] for v in statistic_input_param_RMoE.values()]

    statistic_input_param_RMoE_np = {k: v for k, v in statistic_RMoE_np.items() if any([name in k for name in input_param])}

    norm_RMoE_np = [v['norm'] for v in statistic_input_param_RMoE_np.values()]
    mean_RMoE_np = [v['mean'] for v in statistic_input_param_RMoE_np.values()]
    std_RMoE_np = [v['std'] for v in statistic_input_param_RMoE_np.values()]

    statistic_input_param_RMoE_np_r05 = {k: v for k, v in statistic_RMoE_np_r05.items() if any([name in k for name in input_param])}

    norm_RMoE_np_r05 = [v['norm'] for v in statistic_input_param_RMoE_np_r05.values()]
    mean_RMoE_np_r05 = [v['mean'] for v in statistic_input_param_RMoE_np_r05.values()]
    std_RMoE_np_r05 = [v['std'] for v in statistic_input_param_RMoE_np_r05.values()]


    fig, axs = plt.subplots(1, 3, figsize=(20, 4))
    axs[0].plot(norm_SMoE, label='norm_SMoE')
    axs[0].plot(norm_RMoE, label='norm_RMoE')
    axs[0].plot(norm_RMoE_np, label='norm_RMoE_np')
    axs[0].plot(norm_RMoE_np_r05, label='norm_RMoE_np_r05')
    axs[0].set_title('norm')
    
    axs[1].plot(mean_SMoE, label='mean_SMoE')
    axs[1].plot(mean_RMoE, label='mean_RMoE')
    axs[1].plot(mean_RMoE_np, label='mean_RMoE_np')
    axs[1].plot(mean_RMoE_np_r05, label='mean_RMoE_np_r05')
    axs[1].set_title('mean')
    
    axs[2].plot(std_SMoE, label='std_SMoE')
    axs[2].plot(std_RMoE, label='std_RMoE')
    axs[2].plot(std_RMoE_np, label='std_RMoE_np')
    axs[2].plot(std_RMoE_np_r05, label='std_RMoE_np_r05')
    axs[2].set_title('std')
    fig.legend()
    fig.suptitle(param)
    plt.savefig(f'{SMoE_name}/{param}.png', bbox_inches='tight')
    plt.show()