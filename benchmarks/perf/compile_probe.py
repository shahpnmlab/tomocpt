import sys, time, torch
from tomocpt.networks.swinunetr import MySwinUNETR
from monai.networks.nets.swin_unetr import SwinTransformerBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
mode=sys.argv[1]; ckpt = sys.argv[2]=="ckpt"
torch.manual_seed(0)
m=MySwinUNETR(img_size=64,in_channels=1,out_channels=6,feature_size=36,use_v2=True,use_checkpoint=ckpt,drop_rate=0.3,attn_drop_rate=0.3,dropout_path_rate=0.2).cuda().train()
n=0
if mode=="regional":
    for mod in m.modules():
        if isinstance(mod,(SwinTransformerBlock,UnetrBasicBlock,UnetrUpBlock)):
            mod.compile(); n+=1
opt=torch.optim.RAdam(m.parameters(),1e-4)
x=torch.randn(16,1,64,64,64,device="cuda"); y=torch.rand(16,6,64,64,64,device="cuda")
def step():
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=m(x)[0]
    loss=torch.nn.functional.huber_loss(torch.sigmoid(out.float()),y); opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize(); t=time.perf_counter(); step(); torch.cuda.synchronize(); first=time.perf_counter()-t
for _ in range(4): step()
torch.cuda.synchronize(); t=time.perf_counter()
for _ in range(20): step()
torch.cuda.synchronize(); ms=(time.perf_counter()-t)/20*1000
print(f"RESULT {mode} ckpt={ckpt} compiled_modules={n} first_step={first:.1f}s steady={ms:.0f} ms peak={torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
