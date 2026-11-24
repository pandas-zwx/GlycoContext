"""Sequence-only features. No label, assay intensity or RNA identity as inputs."""
import numpy as np
from .io import site_window, sequence_key


def parse_pairs(structure):
    stacks = {x:[] for x in '([{<'}; pairs = []
    for i,c in enumerate(structure):
        if c in stacks: stacks[c].append(i)
        elif c in ')]}>':
            opener = '([{<'[')]}>'.index(c)]
            if not stacks[opener]: raise ValueError('Unmatched structure bracket')
            pairs.append((stacks[opener].pop(),i))
        elif c != '.': raise ValueError('Unsupported dot-bracket')
    if any(stacks.values()): raise ValueError('Unmatched structure bracket')
    return sorted(pairs)


def graph_arrays(sequence, structure, probabilities):
    n = len(sequence); pairs = parse_pairs(structure)
    probabilities = np.asarray(probabilities,dtype=float)
    if len(structure) != n or probabilities.shape != (len(pairs),) or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Structure/probability mismatch')
    nodes = np.zeros((n,8),dtype=np.float32)
    for i,base in enumerate(sequence): nodes[i,'ACGUN'.index(base)] = 1
    neighbours = np.tile(np.arange(n)[:,None],(1,3)); masks = np.zeros((n,3),dtype=np.float32)
    for i in range(n):
        for slot,j in enumerate((i-1,i+1)):
            if 0 <= j < n: neighbours[i,slot] = j; masks[i,slot] = 1
    confidence = np.zeros(n,dtype=np.float32); partner = np.full(n,-1,dtype=int)
    for (a,b),p in zip(pairs,probabilities):
        if partner[a] >= 0 or partner[b] >= 0: raise ValueError('More than one partner')
        partner[a],partner[b] = b,a
        neighbours[a,2],neighbours[b,2] = b,a; masks[a,2] = masks[b,2] = 1
        confidence[a] = confidence[b] = float(p)
    # The selected model masks the last3 node slots and all7 edge slots.
    return nodes,neighbours,masks,np.zeros((n,3,7),dtype=np.float32),confidence,partner


def cnn_window(sequence, center, partner):
    w = site_window(sequence,center); x = np.zeros((8,51),dtype=np.float32)
    for j,(base,valid) in enumerate(zip(w['sequence'],w['valid_mask'])):
        if not valid: continue
        i = center+j-25; x['ACGUN'.index(base),j] = 1; x[5,j] = 1
        if partner[i] >= 0:
            x[6,j] = 1
            x[7,j] = np.sign(partner[i]-i)*np.log1p(abs(partner[i]-i))/np.log1p(max(len(sequence)-1,1))
    return x


def fold(sequence):
    import RNA
    if RNA.__version__ != '2.7.2': raise ValueError('Reproduction requires ViennaRNA 2.7.2')
    if len(sequence)>2000: raise ValueError('RNA exceeds configured whole-sequence folding cap')
    md = RNA.md(); md.temperature = 37.
    compound = RNA.fold_compound(sequence,md); structure,energy = compound.mfe()
    compound.exp_params_rescale(energy); _,free_energy = compound.pf(); bpp = compound.bpp()
    return dict(sequence_sha256=sequence_key(sequence),structure=structure,
        mfe_kcal_per_mol=float(energy),ensemble_free_energy=float(free_energy),
        RNA_version=RNA.__version__,temperature_C=37,
        probabilities=[float(bpp[a+1][b+1]) for a,b in parse_pairs(structure)])
