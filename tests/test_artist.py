import unittest
import torch
from workers.artist import Studio, Artist, YEARS_STEPS, budget

class ArtistTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.s = Studio()
    def test_brush_and_stop(self):
        c = torch.zeros(2,3,32,32)
        a = torch.tensor([0,self.s.stop])
        out = self.s.paint(c,a)
        self.assertGreater(out[0].sum().item(),0)
        self.assertTrue(torch.equal(out[1],c[1]))
        self.assertEqual(c.sum().item(),0)
    def test_teacher_replays_target(self):
        t, seq = self.s.tasks(8,4)
        c = torch.zeros_like(t)
        for a in seq.T:
            c = self.s.paint(c,a)
        self.assertTrue(torch.equal(c,t))
    def test_reward_prefers_correct_stroke(self):
        c = torch.zeros(1,3,32,32)
        t = self.s.paint(c,torch.tensor([0]))
        self.assertGreater(self.s.reward(c,t,t,torch.tensor([0])).item(),0)
    def test_policy_shapes_and_gradients(self):
        m = Artist(self.s)
        c = torch.zeros(2,3,32,32)
        logits,v = m(c,c)
        self.assertEqual(logits.shape,(2,self.s.stop+1))
        self.assertEqual(v.shape,(2,))
        (logits.mean()+v.mean()).backward()
        self.assertTrue(any(p.grad is not None for p in m.parameters()))
    def test_budget_is_real_actions(self):
        self.assertEqual(budget(5000),5000*YEARS_STEPS)
        self.assertEqual(budget(5000,100),100)
        with self.assertRaises(ValueError): budget(0)
if __name__ == '__main__': unittest.main()
