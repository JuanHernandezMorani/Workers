import unittest
import torch
import workers as w

class WorkersTests(unittest.TestCase):
    def setUp(self):
        self.masks = w.tools(16)
    def test_teacher_reconstructs(self):
        target = w.tasks(self.masks, 12, 3, 4)
        canvas = torch.zeros_like(target)
        for _ in range(3):
            action = w.teacher(canvas, target, self.masks)
            canvas, _, _ = w.step(canvas, target, action, self.masks)
        self.assertTrue(torch.equal(canvas, target))
    def test_reward_and_stop(self):
        target = self.masks[0:1].clone()
        canvas = torch.zeros_like(target)
        out, reward, done = w.step(canvas, target, torch.tensor([0]), self.masks)
        self.assertTrue(torch.equal(out, target))
        self.assertGreater(reward.item(), 0)
        self.assertTrue(done.item())
        out, reward, done = w.step(canvas, target, torch.tensor([len(self.masks)]), self.masks)
        self.assertTrue(torch.equal(out, canvas))
        self.assertTrue(done.item())
        self.assertEqual(reward.item(), 0)
    def test_reproducibility(self):
        self.assertTrue(torch.equal(w.tasks(self.masks, 8, 3, 42), w.tasks(self.masks, 8, 3, 42)))
    def test_training_updates_weights(self):
        torch.manual_seed(3)
        model = w.Policy(16, len(self.masks))
        opt = torch.optim.Adam(model.parameters(), lr=.003)
        target = w.tasks(self.masks, 8, 2, 4)
        before = [p.detach().clone() for p in model.parameters()]
        w.train_batch(model, opt, target, self.masks, 4, True)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
        before = [p.detach().clone() for p in model.parameters()]
        w.train_batch(model, opt, target, self.masks, 4, False)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
    def test_projection(self):
        p = torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        self.assertTrue(torch.allclose(w.project(p, 0), p[:,:2]))
        self.assertTrue(torch.allclose(w.project(p, 90), torch.tensor([[0.,0.],[0.,1.]]), atol=1e-6))

if __name__ == '__main__': unittest.main()
