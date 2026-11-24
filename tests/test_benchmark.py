import unittest
import torch
from workers.artist import Studio, Artist, YEARS_STEPS, block_budget
from workers.benchmark import greedy

class BenchmarkTests(unittest.TestCase):
    def test_resume_finishes_requested_boundary(self):
        block=5000*YEARS_STEPS
        self.assertEqual(block_budget(123,1),block-123)
        self.assertEqual(block_budget(block,2),2*block)
        self.assertEqual(block_budget(123,1,100),100)
    def test_greedy_solves_single_stroke(self):
        s=Studio()
        canvas=torch.zeros(3,3,32,32)
        t=s.paint(canvas,torch.tensor([0,100,900]))
        c,_=greedy(s,t)
        self.assertTrue(torch.equal(c,t))
