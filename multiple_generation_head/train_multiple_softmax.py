import os
import time
from datetime import datetime
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from models.model_gpt2_multi_sofmax import GPT2MultiSoftmax
from utils.dataloader import DataLoaderLite

# distributed
distributed = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if distributed:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"



# model params
model_type = 'gpt2_model_19072.pt'
hook_layers = list(range(7,12))  # 假设是12层模型
freeze_base = True
# train params
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
torch.set_float32_matmul_precision('high')
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16
T = 1024 # sequence length
assert total_batch_size % (B * T * world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
# log params
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
save_steps = 5000 # save a checkpoint every this many steps
assert save_steps % 250 == 0, "save_steps should be divisible by 250 for validation loss evaluation"
run_name = f"gpt2-multi-softmax_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
use_wandb = True
if use_wandb and master_process:
    wandb.init(project="multiple-softmax", name=run_name, config={
        "base_model_type": 'gpt2',
        "total_batch_size": total_batch_size,
        "B": B,
        "T": T,
        "grad_accum_steps": grad_accum_steps,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
    })



# load
model = GPT2MultiSoftmax.from_pretrained(
    model_type,
    hook_layers=hook_layers,
    freeze_base=freeze_base  # 设置为True则只训练新增的层
).to(device)
enc = tiktoken.get_encoding('gpt2')
train_loader = DataLoaderLite(B=B, T=T, process_rank=rank, num_processes=world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=rank, num_processes=world_size, split="val")



# train
import math
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type, master_process=master_process)
if use_compile:
    model = torch.compile(model)
model = DDP(model, device_ids=[local_rank])
raw_model = model.module if distributed else model
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_losses_accum = torch.zeros(raw_model.config.n_layer, device=device)
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss, losses = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
                losses = losses / val_loss_steps
                val_losses_accum += losses.detach()
        if distributed:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_losses_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            if use_wandb:
                log_info = {
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                log_info.update({f"val_loss_layer_{layer}": val_losses_accum[layer].item() for layer in raw_model.hook_layers})
                wandb.log(log_info)
            if step > 0 and (step % save_steps == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"{run_name}_step_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.config.__dict__,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'wandb_id': wandb.run.id if use_wandb else None
                }
                torch.save(checkpoint, checkpoint_path)
        
            if not use_compile:
                model.eval()
                prompt = "Donald Trump is the president of"
                max_new_tokens = 32
                input_ids = [enc.encode(prompt)]
                input_ids = torch.tensor(input_ids, device=device)
                output_list = raw_model.generate(input_ids, max_new_tokens, do_sample=False)
                if use_wandb:
                    samples_table = wandb.Table(columns=["step", "rank", "layer", "text"])
                for layer, output in output_list.items():
                    decoded = enc.decode(output[0].tolist())
                    print(f"Model Layer {layer}:\n{decoded}\n")
                    samples_table.add_data(step, rank, layer, decoded) if use_wandb else None
                if use_wandb:
                    wandb.log({f"{run_name}_generated_samples": samples_table})



    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    losses_accum = torch.zeros(raw_model.config.n_layer, device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if distributed:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss, losses = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        losses = losses / grad_accum_steps
        losses_accum += losses.detach()
        loss.backward()
    if distributed:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(losses_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        if use_wandb:
            log_info = {
                "step": step,
                "train_loss": loss_accum.item(),
                "learning_rate": lr,
                "grad_norm": norm,
                "tokens_per_sec": tokens_per_sec,
            }
            log_info.update({f"loss_layer_{layer}": losses_accum[layer].item() for layer in raw_model.hook_layers})
            wandb.log(log_info)




# clean
if distributed:
    dist.destroy_process_group()
