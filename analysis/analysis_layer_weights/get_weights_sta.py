import torch

params = [ 'attn.proj_query.weight', 'attn.proj_out.weight', 'attn.proj_val.weight', 'attn.proj_key.weight', 'smoe.gate.gate.weight', 'smoe.gate.router_layer.weight', 'smoe.gate.rnn_projector.weight', 'smoe.gate.gate.bias', 'smoe.experts.htoh4.weight', 'smoe.experts.htoh4.bias', 'smoe.experts.h4toh.weight', 'smoe.experts.h4toh.bias', 'smoe.layer_norm.weight', 'smoe.layer_norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'norm3.weight', 'norm3.bias']

n_layer = 32

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


n_layer = 12

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