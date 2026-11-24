"""Checkpoint-compatible GlycoContext. No RNA background or probability calibrator.

The compatibility slots documented below are registered but are not routing
or active physicochemical experts. They are retained to load the supplied
55,352-parameter states without changing their tensor inventory.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class BinnedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(192, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1, bias=False))

    def forward(self, h):
        pieces = [h[:, :, a:b].mean(-1) for a, b in [(0,10),(10,20),(20,31),(31,41),(41,51)]]
        pieces.append(h[:, :, 25])
        return self.project(torch.stack(pieces, -1))


class WindowEncoder(nn.Module):
    def __init__(self, stem, local):
        super().__init__(); self.stem = copy.deepcopy(stem); self.local = copy.deepcopy(local)

    def forward(self, x): return self.local(self.stem(x))


class LocalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Reproduce the reference constructor's initialization RNG, including
        # the temporary flattened head before replacing it with BinnedHead.
        reference = nn.Sequential(nn.Conv1d(6,32,9,padding=4),nn.GELU(),
            nn.Conv1d(32,32,9,padding=4),nn.GELU(),nn.Flatten(),
            nn.Linear(32*51,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        stem = nn.Conv1d(8,32,9,padding=4)
        with torch.no_grad():
            stem.weight.zero_(); stem.weight[:,:6].copy_(reference[0].weight); stem.bias.copy_(reference[0].bias)
        encoder = WindowEncoder(stem, reference[1:4])
        self.cell_bias = nn.Embedding(2,1)
        with torch.no_grad(): self.cell_bias.weight.fill_(float(reference[-1].bias.detach()))
        head = copy.deepcopy(reference[4:]); head[-1] = nn.Linear(64,1,bias=False)
        with torch.no_grad(): head[-1].weight.copy_(reference[-1].weight)
        self.encoder, self.head = encoder, head

    def forward(self, x, cell,condition=True):
        if x.ndim != 3 or x.shape[1:] != (8,51) or cell.shape != (len(x),): raise ValueError('Expected Bx8x51 and B cell IDs')
        if cell.dtype != torch.long or not ((cell >= 0)&(cell < 2)).all(): raise ValueError('Supported cells: HeLa=0, H9=1')
        if not (x[:,5,25] > 0).all(): raise ValueError('Observed center required')
        bias=self.cell_bias(cell).squeeze(-1) if condition else self.cell_bias.weight.mean()
        return self.head(self.encoder(x)).squeeze(-1)+bias


class ProbabilityWeightedGNN(nn.Module):
    def __init__(self):
        super().__init__(); self.depth = 2
        self.register_buffer('feature_mask', torch.tensor([1.,1.,1.,1.,1.,0.,0.,0.]))
        self.input = nn.Linear(8,32)
        self.layers = nn.ModuleList([nn.ModuleDict({'norm':nn.LayerNorm(32),'self':nn.Linear(32,32),
            'back':nn.Linear(32,32,bias=False),'pair':nn.Linear(32,32,bias=False),
            'edge':nn.Linear(7,32,bias=False)}) for _ in range(2)])
        self.readout = nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1,bias=False))
        nn.init.zeros_(self.readout[-1].weight)

    def forward(self, nodes, neighbours, masks, centers, edges):
        valid = nodes[:,:,:5].sum(-1) > 0
        h0 = F.gelu(self.input(nodes*self.feature_mask))*valid[:,:,None]; h = h0
        batch = torch.arange(len(nodes),device=nodes.device)[:,None,None]
        for layer in self.layers:
            z = layer['norm'](h); adjacent = z[batch,neighbours]
            back = (adjacent[:,:,:2]*masks[:,:,:2,None]).sum(2)/masks[:,:,:2].sum(2).clamp_min(1)[:,:,None]
            pair = adjacent[:,:,2]*masks[:,:,2,None]
            # Physicochemical edge slots are inactive in this supplied model.
            e = layer['edge'](edges*0)
            edge = (e[:,:,:2]*masks[:,:,:2,None]).sum(2)/masks[:,:,:2].sum(2).clamp_min(1)[:,:,None]
            message = layer['self'](z)+layer['back'](back)+layer['pair'](pair)+edge
            h = (.9*(h+F.gelu(message)/math.sqrt(2))+.1*h0)*valid[:,:,None]
        center = h[torch.arange(len(h),device=h.device),centers]
        pooled = (h*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        return self.readout(torch.cat([center,pooled*0],-1)).squeeze(-1)


class Core(nn.Module):
    def __init__(self):
        super().__init__(); self.base = LocalCNN(); self.branch = ProbabilityWeightedGNN()
        self.base.head = BinnedHead()


class RepresentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(640),nn.Linear(640,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1,bias=False))
        nn.init.zeros_(self.net[-1].weight)


class SiteModel(nn.Module):
    def __init__(self):
        super().__init__(); self.core = Core(); self.representation = RepresentationHead()
        # Compatibility parameters: not used by the fixed equal-context readout.
        self.global_logits = nn.Parameter(torch.zeros(3))
        self.router = nn.Sequential(nn.Linear(34,8),nn.GELU(),nn.Linear(8,3))
        nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self.readout_delta = nn.Parameter(torch.zeros(32))

    def representation_scores(self, views, cell,condition=True):
        if views.shape != (len(cell),3,640): raise ValueError('Require native/51nt/101nt center embeddings')
        hidden = self.representation.net[:4](views.reshape(-1,640)).reshape(-1,3,32)
        score = self.representation.net[-1](hidden).squeeze(-1).mean(1)
        extra = (hidden.mean(1)*self.readout_delta).sum(-1)*(2*cell-1)
        return score+extra*float(condition), views.new_full((len(cell),3),1/3)


class GlycoContext(nn.Module):
    ABLATIONS = ('full','no_cnn','no_gnn','no_rnafm','no_rnafold','no_cell',
        'no_cnn_structure','no_gnn_pair','inc_cnn_seq','inc_cnn_structure',
        'inc_rnafm','inc_gnn_backbone')

    def __init__(self,ablation='full'):
        super().__init__()
        if ablation not in self.ABLATIONS: raise ValueError('Unknown ablation')
        self.ablation = ablation; self.site = SiteModel()

    def forward(self, x, cell, graph, confidence, return_components=False):
        nodes, neighbours, masks, centers, edges, views = graph[:6]
        incremental = self.ablation.startswith('inc_')
        condition = self.ablation != 'no_cell' and not incremental
        if self.ablation in ('no_rnafold','no_cnn_structure','inc_cnn_seq'):
            x = x.clone(); x[:,6:8] = 0
        cnn = self.site.core.base(x,cell,condition=condition)
        weighted = masks.clone(); weighted[:,:,2] = weighted[:,:,2]*confidence
        if self.ablation in ('no_rnafold','no_gnn_pair','inc_cnn_seq','inc_cnn_structure',
                'inc_rnafm','inc_gnn_backbone'):
            weighted[:,:,2] = 0
        gnn = self.site.core.branch(nodes,neighbours,weighted,centers,edges)
        fm, _ = self.site.representation_scores(views,cell,condition=condition)
        if self.ablation == 'no_cnn': cnn = cnn*0+self.site.core.base.cell_bias(cell).squeeze(-1)
        if self.ablation == 'no_gnn' or self.ablation in ('inc_cnn_seq','inc_cnn_structure','inc_rnafm'):
            gnn = gnn*0
        if self.ablation == 'no_rnafm' or self.ablation in ('inc_cnn_seq','inc_cnn_structure'):
            fm = fm*0
        logits = (cnn+gnn)+fm
        return dict(logits=logits,cnn=cnn,gnn=gnn,rnafm=fm) if return_components else logits
