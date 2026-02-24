import argparse
import os
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from datasets import vqa_imagefolder_dump_attention
from models import InstructBlip_PIP, Blip2_PIP
try:
    from models import Qwen25VL_PIP
except ImportError:
    Qwen25VL_PIP = None
import utils
import shutil

def main(args):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating vqa datasets")

    datasets = vqa_imagefolder_dump_attention(
        image_dir=args.image_dir,
        clean_image_dir=args.clean_image_dir,
        attacked_image_dir=args.attacked_image_dir,
        images_root=args.images_root,
    )

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()               
        sampler = torch.utils.data.DistributedSampler(datasets, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler = None

    data_loader = DataLoader(datasets, batch_size=1, num_workers=4, pin_memory=True, sampler=sampler, shuffle=False, collate_fn=None, drop_last=False)              

    print("Creating model")

    if args.lvlm == "iblip":
        if not args.lvlm_root:
            raise ValueError("--lvlm_root is required for iblip models")
        model = InstructBlip_PIP(args.lvlm_root, args.lvlm_llm)
    elif args.lvlm == "blip2":
        if not args.lvlm_root:
            raise ValueError("--lvlm_root is required for blip2 models")
        model = Blip2_PIP(args.lvlm_root, args.lvlm_llm)
    elif args.lvlm == "qwen25vl":
        if Qwen25VL_PIP is None:
            raise ImportError("Qwen2.5-VL support requires transformers. Install with: pip install transformers")
        model = Qwen25VL_PIP(model_name=args.lvlm_model_id)
    else:
        raise ValueError(f"Unknown lvlm: {args.lvlm}")

    model = model.to(device)   

    print("Start training")
    start_time = time.time()    

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Dumping attention map:"
    print_freq = 5
    
    question_list = json.load(open(args.question_list, "r"))

    layer_num = 0
    head_num = 0

    for n, (image_name, image, is_clean) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        if layer_num==0 and head_num==0:
            image = image.to(device,non_blocking=True)
            layer_num = model.get_attention(image, "Is there a clock?").size(0)
            head_num = model.get_attention(image, "Is there a clock?").size(1)
        attention_map = torch.zeros(len(question_list), layer_num, head_num, 32)
        image = image.to(device,non_blocking=True)
        for i, question in enumerate(question_list):
            attention_map[i] = model.get_attention(image, question)

        rel_image_path = image_name[0]
        rel_no_ext = os.path.splitext(rel_image_path)[0]
        out_path = os.path.join(args.output_attention_dir, f"{rel_no_ext}.npy")
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        np.save(out_path, attention_map.numpy())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    
    # default config
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    # model config
    parser.add_argument('--lvlm_root', default=None, type=str, help='Local model root (required for blip2/iblip, optional for qwen25vl)')
    parser.add_argument('--lvlm', required=True, type=str, choices=["blip2_flan-t5-xl", "blip2_flan-t5-xxl", "blip2_opt-2.7b", "blip2_opt-6.7b", "iblip_flan-t5-xl", "iblip_flan-t5-xxl", "iblip_vicuna-7b", "iblip_vicuna-13b", "qwen25vl"])
    parser.add_argument('--lvlm_model_id', default="Qwen/Qwen2.5-VL-3B-Instruct", type=str, help='HuggingFace model id for qwen25vl (e.g. Qwen/Qwen2.5-VL-3B-Instruct)')
 
    # dataset config
    parser.add_argument('--image_dir', default=None, type=str, help='Legacy image root containing an image/ subfolder')
    parser.add_argument('--clean_image_dir', default=None, type=str, help='Directory containing clean images')
    parser.add_argument('--attacked_image_dir', default=None, type=str, help='Directory containing attacked images')
    parser.add_argument('--images_root', default=None, type=str, help='Recursively dump all images under this root, preserving relative folder structure')
    parser.add_argument('--question_list', required=True, type=str)
    parser.add_argument('--output_attention_dir', default=None, type=str, help='Output directory for dumped attention maps')
    
    args = parser.parse_args()

    if args.lvlm == "qwen25vl":
        args.lvlm_llm = getattr(args, "lvlm_model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    else:
        args.lvlm, args.lvlm_llm = args.lvlm.split("_", 1)

    if args.image_dir is None and args.clean_image_dir is None and args.attacked_image_dir is None and args.images_root is None:
        raise ValueError("Provide --images_root, --image_dir, or at least one of --clean_image_dir / --attacked_image_dir")

    if args.output_attention_dir is None:
        if args.image_dir is not None:
            args.output_attention_dir = os.path.join(args.image_dir, "attention_map_index0")
        else:
            raise ValueError("--output_attention_dir is required when --image_dir is not provided")

    Path(args.output_attention_dir).mkdir(parents=True, exist_ok=True)
    if args.images_root is not None:
        main(args)
    elif args.image_dir is not None:
        if os.path.exists(os.path.join(args.image_dir, "vqa_result.json")):
            shutil.copy(args.question_list, os.path.join(args.image_dir, "question_list.json"))
            main(args)
        else:
            print(f"Not finish running in {args.image_dir}")
    else:
        main(args)
