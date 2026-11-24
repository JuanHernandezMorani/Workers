import tempfile
import unittest
from pathlib import Path
import torch
from workers.artist import Studio, Artist
from workers.train import evaluate, reinforce, save, restore, draw

class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.s=Studio()
        self.m=Artist(self.s)
        self.o=torch.optim.Adam(self.m.parameters(),lr=.001)
    def test_reinforcement_counts_real_transitions(self):
        old={k:v.clone() for k,v in self.m.state_dict().items()}
        stats=reinforce(self.s,self.m,self.o,steps=19,envs=4)
        self.assertEqual(stats["actions"],19)
        self.assertTrue(any(not torch.equal(v,old[k]) for k,v in self.m.state_dict().items()))
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"latest.pt"
            save(p,self.m,self.o,{"actions":19,"supervised_updates":3})
            other=Artist(self.s)
            opt=torch.optim.Adam(other.parameters())
            state=restore(p,other,opt)
            self.assertEqual(state["actions"],19)
            self.assertTrue(all(torch.equal(v,other.state_dict()[k]) for k,v in self.m.state_dict().items()))
    def test_eval_does_not_change_rng(self):
        state=torch.get_rng_state().clone()
        a=evaluate(self.s,self.m,n=4)
        b=evaluate(self.s,self.m,n=4)
        self.assertEqual(a,b)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_draw_replay(self):
        target,_=self.s.tasks(3)
        canvas,seq=draw(self.s,self.m,target)
        replay=torch.zeros_like(target)
        for action in seq.T: replay=self.s.paint(replay,action)
        self.assertTrue(torch.equal(canvas,replay))
