import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.io import site_window
from glycocontext.features import graph_arrays,cnn_window
from glycocontext.model import GlycoContext
from glycocontext.data import roles


class CoreTests(unittest.TestCase):
    def test_padding_is_not_observed_N(self):
        w=site_window('NACGU',0)
        self.assertEqual(w['sequence'][24:27],'NNA');self.assertFalse(w['valid_mask'][24]);self.assertTrue(w['valid_mask'][25])

    def test_graph_and_probability_alignment(self):
        values=graph_arrays('ACGU','(..)',[.37])
        nodes,nb,mask,edges,confidence,partner=values
        np.testing.assert_array_equal(nb[[0,3],2],[3,0]);np.testing.assert_allclose(confidence[[0,3]],[.37,.37])
        self.assertEqual(cnn_window('ACGU',0,partner).shape,(8,51))
        self.assertEqual(float(mask[0,0]),0)

    def inputs(self):
        values=graph_arrays('ACGUACGU','((....))',[.7,.3]);nodes,nb,mask,edges,confidence,partner=values
        x=torch.tensor(np.stack([cnn_window('ACGUACGU',3,partner)]*2))
        graph=tuple(torch.tensor(np.stack([v]*2)) for v in [nodes,nb,mask])
        graph=graph+(torch.tensor([3,3]),torch.tensor(np.stack([edges]*2)),torch.randn(1,3,640).repeat(2,1,1))
        return x,torch.tensor([0,1]),graph,torch.tensor(np.stack([confidence]*2))

    def test_inventory_and_cell_ablation(self):
        model=GlycoContext('no_cell').eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),55352)
        with torch.no_grad():
            model.site.readout_delta.normal_();model.site.core.base.cell_bias.weight.copy_(torch.tensor([[1.],[-2.]]))
            value=model(*self.inputs())
        torch.testing.assert_close(value[0],value[1],rtol=0,atol=0)

    def test_ablation_inventory_and_backward(self):
        for arm in GlycoContext.ABLATIONS:
            torch.manual_seed(0);model=GlycoContext(arm)
            z=model(*self.inputs());self.assertEqual(tuple(z.shape),(2,));z.square().mean().backward()
            self.assertTrue(torch.isfinite(z).all())

    def test_fine_structure_ablation_paths(self):
        args=self.inputs();x,cell,graph,confidence=args
        model=GlycoContext('full').eval()
        with torch.no_grad():
            full=model(*args,return_components=True)
            model.ablation='no_cnn_structure'
            no_cnn_structure=model(*args,return_components=True)
            model.ablation='full';zero_x=x.clone();zero_x[:,6:8]=0
            cnn_reference=model(zero_x,cell,graph,confidence,return_components=True)
            model.ablation='no_gnn_pair'
            no_gnn_pair=model(*args,return_components=True)
            model.ablation='full'
            pair_reference=model(x,cell,graph,torch.zeros_like(confidence),return_components=True)
        torch.testing.assert_close(no_cnn_structure['cnn'],cnn_reference['cnn'])
        torch.testing.assert_close(no_cnn_structure['gnn'],full['gnn'])
        torch.testing.assert_close(no_gnn_pair['cnn'],full['cnn'])
        torch.testing.assert_close(no_gnn_pair['gnn'],pair_reference['gnn'])

    def test_incremental_component_path(self):
        args=self.inputs();model=GlycoContext('inc_cnn_seq').eval()
        with torch.no_grad():
            outputs={}
            for arm in ('inc_cnn_seq','inc_cnn_structure','inc_rnafm','inc_gnn_backbone','no_cell','full'):
                model.ablation=arm;outputs[arm]=model(*args,return_components=True)
        zero=torch.zeros_like(outputs['full']['gnn'])
        torch.testing.assert_close(outputs['inc_cnn_seq']['gnn'],zero)
        torch.testing.assert_close(outputs['inc_cnn_seq']['rnafm'],zero)
        torch.testing.assert_close(outputs['inc_cnn_structure']['gnn'],zero)
        torch.testing.assert_close(outputs['inc_cnn_structure']['rnafm'],zero)
        torch.testing.assert_close(outputs['inc_rnafm']['gnn'],zero)
        torch.testing.assert_close(outputs['inc_rnafm']['rnafm'],outputs['inc_gnn_backbone']['rnafm'])
        torch.testing.assert_close(outputs['inc_gnn_backbone']['cnn'],outputs['no_cell']['cnn'])
        torch.testing.assert_close(outputs['inc_gnn_backbone']['rnafm'],outputs['no_cell']['rnafm'])

    def test_group_cannot_cross_roles(self):
        records=[dict(candidate_id=str(i),group='shared',sequence_sha256=str(i)) for i in range(2)]
        with self.assertRaises(ValueError):roles(records,dict(E=['0'],S=['1'],C=[],test=[]))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
