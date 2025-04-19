"""For use with gpu_parallel.py, see README.md"""
import argparse
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpu_parallel import get_worker_rank, init_worker_logger, TaskQueue

if "NV_YT_OPERATION_ID" in os.environ:
    import nirvana_dl


def parse_args():
    parser = argparse.ArgumentParser(description="Eval baselines")
    parser.add_argument(
        "--queue",
        type=str,
        default=None,
        help="Endpoint for a zmq task dispenser that dispenses task indices. Provide *either* this or start & end"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First task to be processed by script inclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last task to be processed by script exclusive. E.g --start 0 --end 100 will process tasks [0-99]"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default='.',
        help='Results will be written to "args.eval_folder/evals_data/limo/exp_name". If running on Nirvana, use $SNAPSHOT_PATH'
    )
    parser.add_argument(
        "--dump_snapshot_freq",
        type=int,
        default=4,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank = get_worker_rank()
    device = torch.device('cuda')  # gpu_parallel already sets CUDA_VISIBLE_DEVICES for you
    logger = init_worker_logger()
    logger.info(f'The script was run in the following way:')
    logger.info(f"python {__file__} \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))
    logger.info(f'IN_NIRVANA = {"NV_YT_OPERATION_ID" in dict(os.environ)}')
    logger.info(f'Output directory: {args.save_folder}')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
        logger.info(f'Created directory {args.save_folder}')
    else:
        logger.info(f'Directory {args.save_folder} already exists')

    logger.info('Loading model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map=device, torch_dtype='auto')
    dataset = load_dataset("GAIR/LIMO", split="train")  # <-- load the entire dataset for each worker to handle idx correctly
    local_tasks_solved = 0

    def _run_task(idx: int):
        nonlocal local_tasks_solved
        task_output_path = f'{args.save_folder}/Task_{idx}.txt'
        if os.path.exists(task_output_path):
            return  # already solved by previous attempt and saved in snapshot
        local_tasks_solved += 1

        ######### EXAMPLE CODE ###########
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        prompt_str = system_prompt + str(dataset[idx]['question'])
        prompt_with_template_str = tokenizer.apply_chat_template([dict(role='user', content=prompt_str)], add_generation_prompt=True, tokenize=False)
        prompt = torch.tensor(tokenizer.encode(prompt_with_template_str, add_special_tokens=False)).to(device)[None, :]
        response = model.generate(prompt, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id)
        with open(task_output_path, 'w') as file:
            file.write(tokenizer.decode(response[0]))
        logger.info(f"Finished task {idx=}")   # maybe add some statistics here
        ######### END OF EXAMPLE CODE ###########

        if "NV_YT_OPERATION_ID" in os.environ and rank == 0 and (
                local_tasks_solved % args.dump_snapshot_freq == args.dump_snapshot_freq - 1):
            nirvana_dl.snapshot.dump_snapshot()   # note: gpu_parallel will also re-dump snapshot at the end
            logger.info("Dumped Nirvana snapshot")

    if args.start is not None and args.end is not None:
        logger.info(f'Generating tasks [{args.start}; {args.end})')
        for idx in tqdm(range(args.start, args.end), desc=f'Process {rank}'):
            _run_task(idx)
    elif args.queue is not None:
        logger.info(f'Generating tasks from {args.queue}')
        for idx in tqdm(TaskQueue.iterate_tasks_from_queue(endpoint=args.queue), desc=f"Process {rank}"):
            _run_task(idx)
    else:
        raise NotImplementedError("Please specify either --queue or both --start and --end")
    logger.info(f'Process {rank} has finished.')


if __name__ == "__main__":
    main()
