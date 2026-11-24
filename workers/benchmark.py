"""Independent seeded holdout and deterministic immediate-gain baseline."""
import argparse
import json
import time
from pathlib import Path
import torch
from .artist import Studio,Artist
from .train import draw,restore

@torch.no_grad()
def greedy(studio,target,limit=8):
    canvas=torch.zeros_like(target)
    seq=[]
    masks=studio.masks[:-1].reshape(5,4,64,-1)[:,0].reshape(320,-1).float()
    colors=studio.inks[:-1].reshape(5,4,64,3,1,1)[0,:,0]
    for _ in range(limit):
        old=(canvas-target).square().sum(1)
        new=(colors[None]-target[:,None]).square().sum(2)
        gain=(old[:,None]-new).flatten(2)@masks.T
        gain=gain.reshape(len(target),4,5,64).transpose(1,2).reshape(len(target),-1)
        gain/=target.square().sum((1,2,3)).clamp_min(1)[:,None]
        val,a=gain.max(1)
        a=torch.where(val>.005,a,studio.stop)
        canvas=studio.paint(canvas,a); seq.append(a)
        if (a==studio.stop).all(): break
    return canvas,torch.stack(seq,1)

def metrics(s,c,t,seq):
    err=s.error(c,t); a,b=c.amax(1)>0,t.amax(1)>0
    return {"score":float((1-err).mean()),"iou":float(((a&b).sum((1,2))/(a|b).sum((1,2)).clamp_min(1)).mean()),
            "exact":float((err<1e-6).float().mean()),"mean_strokes":float((seq!=s.stop).sum(1).float().mean())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--samples",type=int,default=512)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if min(args.samples,args.threads)<=0:p.error("positive samples and threads required")
    torch.set_num_threads(args.threads)
    s=Studio(); m=Artist(s); state=restore(args.checkpoint,m)
    torch.manual_seed(88003)
    result={"seed":88003,"samples_per_group":args.samples,"parameters":sum(p.numel() for p in m.parameters()),"training_state":state,"groups":{}}
    for strokes in [4,8]:
        t,_=s.tasks(args.samples,strokes)
        group={}
        for name,fn in [("neural",lambda:draw(s,m,t)),("greedy",lambda:greedy(s,t))]:
            start=time.perf_counter(); c,seq=fn()
            group[name]={**metrics(s,c,t,seq),"seconds":time.perf_counter()-start}
        result["groups"][str(strokes)]=group
    output=json.dumps(result,indent=2)
    print(output)
    if args.output:args.output.write_text(output+"\n",encoding="utf-8")
if __name__=="__main__": main()
