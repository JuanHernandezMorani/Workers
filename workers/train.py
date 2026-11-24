"""CPU-first imitation and on-policy actor-critic training."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from PIL import Image
from .artist import Studio, Artist, YEARS_STEPS, budget, block_budget

@torch.no_grad()
def draw(studio,model,target,limit=8):
    canvas=torch.zeros_like(target)
    done=torch.zeros(len(target),dtype=torch.bool)
    seq=[]
    for _ in range(limit):
        logits,_=model(canvas,target)
        a=torch.where(done,studio.stop,logits.argmax(1))
        canvas=studio.paint(canvas,a)
        seq.append(a)
        done |= a==studio.stop
        if done.all(): break
    return canvas,torch.stack(seq,1)

@torch.no_grad()
def evaluate(studio,model,n=128,seed=77001):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        t,_=studio.tasks(n,4)
        c,seq=draw(studio,model,t)
        err=studio.error(c,t)
        a,b=c.amax(1)>0,t.amax(1)>0
        iou=(a&b).sum((1,2))/(a|b).sum((1,2)).clamp_min(1)
        return {"score":float((1-err).mean()),"iou":float(iou.mean()),
                "exact":float((err<1e-6).float().mean()),
                "mean_strokes":float((seq!=studio.stop).sum(1).float().mean())}

def supervise(studio,model,opt,updates,batch=64):
    loss=0.
    for i in range(updates):
        c,t,a=studio.demonstration(batch,1+min(3,i*4//max(updates,1)))
        logits,_=model(c,t)
        l=nn.functional.cross_entropy(logits,a)
        opt.zero_grad(set_to_none=True)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        loss+=l.item()
    return {"updates":updates,"examples":updates*batch,"loss":loss/max(1,updates)}

def reinforce(studio,model,opt,steps,envs=16,horizon=8):
    if min(steps,envs,horizon)<=0: raise ValueError("Positive rollout arguments required")
    used=0
    reward_sum=0.
    updates=0
    while used<steps:
        n=min(envs,steps-used)
        h=min(horizon,(steps-used)//n)
        t,_=studio.tasks(n,4)
        c=torch.zeros_like(t)
        logs,values,entropies,rewards,dones=[],[],[],[],[]
        for i in range(h):
            logits,v=model(c,t)
            dist=torch.distributions.Categorical(logits=logits)
            a=dist.sample()
            after=studio.paint(c,a)
            r=studio.reward(c,after,t,a)
            done=(a==studio.stop)|(i==h-1)
            logs.append(dist.log_prob(a)); values.append(v)
            entropies.append(dist.entropy()); rewards.append(r); dones.append(done)
            reward_sum+=r.sum().item()
            used+=n
            c=after
            if i+1<h and done.any():
                new,_=studio.tasks(int(done.sum()),4)
                t=t.clone(); c=c.clone()
                t[done]=new; c[done]=0.
        returns=[]
        g=torch.zeros(n)
        for r,d in zip(reversed(rewards),reversed(dones)):
            g=r+.99*g*(~d)
            returns.append(g)
        ret=torch.stack(returns[::-1])
        val=torch.stack(values)
        adv=ret-val.detach()
        loss=-(torch.stack(logs)*adv).mean()+.5*(val-ret).square().mean()-.003*torch.stack(entropies).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        updates+=1
    return {"actions":used,"updates":updates,"reward_per_action":reward_sum/used}

def save(path,model,opt,state):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    torch.save({"version":1,"model":model.state_dict(),"optimizer":opt.state_dict(),
                "state":state,"rng":torch.get_rng_state()},tmp)
    tmp.replace(path)

def restore(path,model,opt=None):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload["version"]!=1: raise ValueError("Unsupported checkpoint version")
    model.load_state_dict(payload["model"])
    if opt is not None: opt.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"])
    return payload["state"]

def image(t):
    return Image.fromarray((t.permute(1,2,0).clamp(0,1)*255).byte().numpy())

@torch.no_grad()
def examples(studio,model,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(77002)
        targets,_=studio.tasks(12,4)
        result,seq=draw(studio,model,targets)
        sheet=Image.new("RGB",(12*64,128),"#202020")
        for i,(t,c) in enumerate(zip(targets,result)):
            sheet.paste(image(t).resize((64,64),Image.Resampling.NEAREST),(i*64,0))
            sheet.paste(image(c).resize((64,64),Image.Resampling.NEAREST),(i*64,64))
        sheet.save(out/"examples.png")
        (out/"strokes.json").write_text(json.dumps([[studio.describe(a) for a in row.tolist()] for row in seq],indent=2),encoding="utf-8")

def report(out,record):
    with (out/"history.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
    print(json.dumps(record),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["supervise","reinforce","evaluate","draw"])
    p.add_argument("--out",type=Path,default=Path("runs/artist"))
    p.add_argument("--updates",type=int,default=250)
    p.add_argument("--batch",type=int,default=48)
    p.add_argument("--steps",type=int)
    p.add_argument("--years",type=int)
    p.add_argument("--blocks",type=int,default=1)
    p.add_argument("--envs",type=int,default=16)
    p.add_argument("--chunk",type=int,default=4096)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--seed",type=int,default=123)
    p.add_argument("--input",type=Path)
    args=p.parse_args()
    if min(args.updates,args.batch,args.envs,args.chunk,args.threads)<=0: p.error("positive arguments required")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    s=Studio(); m=Artist(s); opt=torch.optim.Adam(m.parameters(),lr=.001)
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    latest=out/"latest.pt"; best=out/"best.pt"
    state={"actions":0,"supervised_updates":0,"best_score":-1e9}
    if latest.exists(): state=restore(latest,m,opt)
    elif args.mode!="supervise": p.error("Train a supervised checkpoint first")
    if args.mode=="draw":
        if not args.input: p.error("--input required")
        if best.exists(): restore(best,m)
        import numpy as np
        t=torch.from_numpy(np.array(Image.open(args.input).convert("RGB").resize((32,32)))).permute(2,0,1).float()[None]/255
        c,seq=draw(s,m,t)
        image(c[0]).resize((512,512),Image.Resampling.NEAREST).save(out/"drawing.png")
        (out/"drawing.json").write_text(json.dumps([s.describe(a) for a in seq[0].tolist()],indent=2),encoding="utf-8")
        return
    if args.mode=="evaluate":
        if best.exists(): restore(best,m)
        print(json.dumps(evaluate(s,m))); examples(s,m,out); return
    def checkpoint(kind,stats,elapsed):
        val=evaluate(s,m)
        rec={"phase":kind,**stats,**val,"elapsed_seconds":elapsed,
             "rl_actions_total":state["actions"],"equivalent_years":state["actions"]/YEARS_STEPS,
             "completed_5000_year_blocks":state["actions"]//(5000*YEARS_STEPS)}
        if val["score"]>state["best_score"]:
            state["best_score"]=val["score"]
            save(best,m,opt,state)
            examples(s,m,out)
        save(latest,m,opt,state)
        report(out,rec)
    if args.mode=="supervise":
        report(out,{"phase":"before_supervision",**evaluate(s,m)})
        start=time.perf_counter()
        stats=supervise(s,m,opt,args.updates,args.batch)
        state["supervised_updates"]+=args.updates
        checkpoint("supervised",stats,time.perf_counter()-start)
    else:
        for group in opt.param_groups: group["lr"]=.0002
        remaining=budget(args.years,args.steps) if args.years is not None else block_budget(state["actions"],args.blocks,args.steps)
        block=5000*YEARS_STEPS
        while remaining:
            n=min(remaining,args.chunk,block-state["actions"]%block)
            start=time.perf_counter()
            stats=reinforce(s,m,opt,n,args.envs)
            elapsed=time.perf_counter()-start
            state["actions"]+=stats["actions"]; remaining-=stats["actions"]
            checkpoint("reinforcement",stats,elapsed)
            if state["actions"]%block==0:
                save(out/("block_%06d.pt"%(state["actions"]//block)),m,opt,state)
if __name__=="__main__": main()
