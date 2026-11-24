"""Workers V0: vectorized tool-action learning, not pixel generation."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical


def tools(size):
    if size < 4:
        raise ValueError('size must be >= 4')
    eye = torch.eye(size, dtype=torch.bool)
    rows = eye[:, :, None].expand(-1, -1, size)
    cols = rows.transpose(1, 2)
    return torch.cat((rows, cols, eye[None], eye.flip(1)[None])).flatten(1).float()


def tasks(masks, count, strokes, seed):
    if not 1 <= strokes <= len(masks):
        raise ValueError('invalid stroke count')
    rng = torch.Generator().manual_seed(seed)
    ids = torch.rand(count, len(masks), generator=rng).topk(strokes, dim=1).indices
    return masks[ids].amax(dim=1)


def teacher(canvas, target, masks):
    gains = (target * (1-canvas)).matmul(masks.T) - ((1-target)*(1-canvas)).matmul(masks.T)
    value, action = gains.max(dim=1)
    return torch.where(value > 0, action, len(masks))


def step(canvas, target, action, masks):
    bank = torch.cat((masks, torch.zeros_like(masks[:1])))
    out = torch.maximum(canvas, bank[action])
    old = (canvas != target).sum(dim=1).float()
    new = (out != target).sum(dim=1).float()
    reward = (old-new)/math.sqrt(canvas.shape[1])
    done = (new == 0) | (action == len(masks))
    return out, reward, done


def project(points, yaw, pitch=0, focal=None, distance=4):
    """Rotate shared 3D points; orthographic or pinhole perspective."""
    a,b = math.radians(yaw),math.radians(pitch)
    x,y,z = points.unbind(-1)
    u,w = x*math.cos(a)+z*math.sin(a), -x*math.sin(a)+z*math.cos(a)
    v,d = y*math.cos(b)-w*math.sin(b), y*math.sin(b)+w*math.cos(b)
    xy = torch.stack((u,v),dim=-1)
    if focal is None:
        return xy
    depth = distance+d
    if (depth <= 0).any():
        raise ValueError('point behind camera')
    return xy*focal/depth[...,None]


class Policy(nn.Module):
    def __init__(self, size, actions):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2*size*size,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.actor = nn.Linear(128,actions+1)
        self.critic = nn.Linear(128,1)

    def forward(self, canvas, target):
        h = self.body(torch.cat((canvas,target),dim=-1))
        return self.actor(h), self.critic(h).squeeze(-1)


def train_batch(model, opt, target, masks, horizon, supervised):
    canvas = torch.zeros_like(target)
    active = torch.ones(len(target),dtype=torch.bool)
    records = []
    for _ in range(horizon):
        logits, value = model(canvas,target)
        dist = Categorical(logits=logits)
        action = teacher(canvas,target,masks) if supervised else dist.sample()
        out,reward,done = step(canvas,target,action,masks)
        records.append((dist.log_prob(action),value,reward,active,dist.entropy()))
        canvas = torch.where(active[:,None],out,canvas)
        active = active & ~done
        if not active.any():
            break
    ret = torch.zeros(len(target))
    loss = torch.zeros(())
    count = 0
    for logp,value,reward,valid,entropy in reversed(records):
        ret = torch.where(valid,reward+.99*ret,0.)
        advantage = ret-value
        term = -logp if supervised else -logp*advantage.detach()+.5*advantage.square()-.01*entropy
        loss = loss + term[valid].sum()
        count += int(valid.sum())
    opt.zero_grad()
    (loss/count).backward()
    nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()
    return count


@torch.no_grad()
def evaluate(model, target, masks, horizon, random=False, seed=0):
    canvas = torch.zeros_like(target)
    active = torch.ones(len(target),dtype=torch.bool)
    rng = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(horizon):
        action = torch.randint(len(masks)+1,(len(target),),generator=rng) if random else model(canvas,target)[0].argmax(dim=1)
        action = torch.where(active,action,len(masks))
        canvas,_,done = step(canvas,target,action,masks)
        history.append(action)
        active = active & ~done
    iou = (canvas*target).sum(1)/(canvas+target).clamp(max=1).sum(1).clamp(min=1)
    return {'iou':float(iou.mean()),'exact':float((canvas==target).all(1).float().mean())},canvas,torch.stack(history,dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',type=int,default=16)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--supervised',type=int,default=500)
    p.add_argument('--rl',type=int,default=200)
    p.add_argument('--strokes',type=int,default=3)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--out',type=Path,default=Path('runs/v0'))
    a = p.parse_args()
    if min(a.batch,a.threads,a.strokes)<1 or min(a.supervised,a.rl,a.seed)<0:
        p.error('invalid training arguments')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    masks = tools(a.size)
    model = Policy(a.size,len(masks))
    opt = torch.optim.Adam(model.parameters(),lr=.003)
    horizon = a.strokes+2
    target = tasks(masks,512,a.strokes,2**40+a.seed)
    report = {'config':{**vars(a),'out':str(a.out)},'parameters':sum(p.numel() for p in model.parameters()),'random':evaluate(model,target,masks,horizon,True,a.seed)[0],'untrained':evaluate(model,target,masks,horizon)[0]}
    n = 0
    for label,iterations,supervised in [('supervised',a.supervised,True),('reinforcement',a.rl,False)]:
        if not supervised:
            opt = torch.optim.Adam(model.parameters(),lr=.0003)
        start = time.perf_counter()
        count = 0
        for _ in range(iterations):
            count += train_batch(model,opt,tasks(masks,a.batch,a.strokes,a.seed+n),masks,horizon,supervised)
            n += 1
        duration = time.perf_counter()-start
        report[label] = {**evaluate(model,target,masks,horizon)[0],'actions':count,'seconds':duration,'actions_per_second':count/max(duration,1e-9)}
        a.out.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.state_dict(),'size':a.size,'actions':len(masks)},a.out/(label+'.pt'))
    _,canvas,history = evaluate(model,target,masks,horizon)
    (a.out/'examples.json').write_text(json.dumps({'target':target[:8].tolist(),'canvas':canvas[:8].tolist(),'actions':history[:8].tolist()}))
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
