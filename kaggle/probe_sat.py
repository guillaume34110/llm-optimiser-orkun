import sys, random, types
from pathlib import Path
from orkun.eval.minibench import _build
from orkun.policy.group_rollout import GroupRollout
from orkun.goals.families import ALL_FAMILIES

ckpt = sys.argv[1]
args = types.SimpleNamespace(
    checkpoint=ckpt,
    config="/root/Orkish/configs/waaagh_grot_80m_v11.yaml",
    orkish_repo="/root/Orkish",
    device="cuda",
)
net, tok, ktools = _build(args)
fam = [f for f in ALL_FAMILIES if f.name == "sat_batch"][0]

greedy = GroupRollout(net, tok, temperature=0.0, top_p=1.0, max_new=256, known_tools=ktools, base_seed=0)
sampler = GroupRollout(net, tok, temperature=0.45, top_p=0.95, max_new=256, known_tools=ktools, base_seed=0)

print("ckpt:", ckpt)
for d in (0, 1, 2):
    gg, sb, npass = [], [], 0
    sample_txt = None
    for s in range(8):
        t = fam.sample(random.Random(5000 + s), d)
        g = greedy.rollout_group(t, samples=1, seed=s).samples[0]
        gg.append(g.graded)
        if sample_txt is None:
            sample_txt = g.assistant_text[:220]
        r = sampler.rollout_group(t, samples=4, seed=s)
        sb.append(max(x.graded for x in r.samples))
        npass += int(any(x.passed for x in r.samples))
    print(f"d{d}: greedy graded avg={sum(gg)/len(gg):.3f}  sampled best@4 graded avg={sum(sb)/len(sb):.3f}  sampled pass={npass}/8")
    print(f"   sample greedy text: {sample_txt!r}")
