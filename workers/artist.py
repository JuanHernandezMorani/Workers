"""Headless tool-based raster drawing environment and compact policy."""
import torch
from torch import nn

YEARS_STEPS = 8 * 3600 * 250

def budget(years, cap=None):
    if years <= 0 or (cap is not None and cap <= 0):
        raise ValueError("Positive years and action cap required")
    total = years * YEARS_STEPS
    return min(total,cap) if cap is not None else total

class Studio:
    size = 32
    grid = 8
    tools = ("dot","disk","horizontal","vertical","diagonal")
    colors = ("ivory","teal","amber","erase")
    def __init__(self):
        y,x = torch.meshgrid(torch.arange(32),torch.arange(32),indexing="ij")
        cy,cx = torch.meshgrid(torch.arange(8)*4+2,torch.arange(8)*4+2,indexing="ij")
        dx = x[None] - cx.flatten()[:,None,None]
        dy = y[None] - cy.flatten()[:,None,None]
        shapes = torch.stack([
            dx.square()+dy.square()<=4,
            dx.square()+dy.square()<=25,
            (dx.abs()<=5)&(dy.abs()<=1),
            (dy.abs()<=5)&(dx.abs()<=1),
            ((dx-dy).abs()<=1)&(dx.abs()<=4)&(dy.abs()<=4)])
        masks = shapes[:,None].expand(-1,4,-1,-1,-1).reshape(-1,1,32,32)
        palette = torch.tensor([[.91,.86,.72],[.15,.72,.60],[.92,.52,.16],[0.,0.,0.]])
        colors = palette[None,:,None].expand(5,-1,64,-1).reshape(-1,3,1,1)
        self.stop = len(masks)
        self.masks = torch.cat([masks,torch.zeros(1,1,32,32,dtype=torch.bool)])
        self.inks = torch.cat([colors,torch.zeros(1,3,1,1)])
        self.positive = torch.arange(self.stop).reshape(5,4,64)[:,:3].flatten()
    def paint(self,canvas,actions):
        return torch.where(self.masks[actions],self.inks[actions],canvas)
    def tasks(self,n,max_strokes=4):
        if not 1<=max_strokes<=8: raise ValueError("max_strokes must be 1..8")
        lengths = torch.randint(1,max_strokes+1,(n,))
        seq = self.positive[torch.randint(len(self.positive),(n,max_strokes))]
        seq = torch.where(torch.arange(max_strokes)[None]<lengths[:,None],seq,self.stop)
        target = torch.zeros(n,3,32,32)
        for a in seq.T: target = self.paint(target,a)
        return target,seq
    @staticmethod
    def error(canvas,target):
        return (canvas-target).square().sum((1,2,3))/target.square().sum((1,2,3)).clamp_min(1.)
    def reward(self,before,after,target,action):
        return self.error(before,target)-self.error(after,target)-.005*(action!=self.stop)
    def demonstration(self,n,max_strokes=4):
        target,seq = self.tasks(n,max_strokes)
        lengths = (seq!=self.stop).sum(1)
        prefix = (torch.rand(n)*(lengths+1)).long()
        canvas = torch.zeros_like(target)
        for i,a in enumerate(seq.T):
            canvas = self.paint(canvas,torch.where(prefix>i,a,self.stop))
        seq = torch.cat([seq,torch.full((n,1),self.stop)],1)
        return canvas,target,seq[torch.arange(n),prefix]
    def describe(self,a):
        if a==self.stop: return {"tool":"stop"}
        tool,rem=divmod(int(a),4*64)
        color,loc=divmod(rem,64)
        y,x=divmod(loc,8)
        return {"tool":self.tools[tool],"color":self.colors[color],"x":x*4+2,"y":y*4+2}

class Artist(nn.Module):
    def __init__(self,studio):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(9,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,48,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(48,64,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU())
        self.brush=nn.Conv2d(64,studio.stop//64,1)
        self.global_head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,canvas,target):
        f=self.features(torch.cat([canvas,target,target-canvas],1))
        g=self.global_head(f.mean((2,3)))
        return torch.cat([self.brush(f).flatten(1),g[:,:1]],1),g[:,1]

def block_budget(actions,blocks=1,cap=None):
    if actions<0 or blocks<=0 or (cap is not None and cap<=0):
        raise ValueError("Invalid block budget")
    block=5000*YEARS_STEPS
    total=blocks*block-actions%block
    return min(total,cap) if cap is not None else total
