"""
fineweb-edu dataset (for gpt2 pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

downloads and tokenizes the data, saving shards to ./edu_fineweb10B as uint16 .npy.
shard 0 becomes the val split, every later shard is train, which is what
DataLoaderLite in main.py expects.

  uv run fineweb.py                # full sample-10BT: 100 shards, ~20gb on disk
  uv run fineweb.py --shards 2     # first 2 shards (200m tokens), enough to train

the dataset is streamed, so only the parquet files we actually consume get pulled.
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # delimits documents


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def shard_path(data_cache_dir, shard_index):
    split = "val" if shard_index == 0 else "train"
    return os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:06d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=None,
                        help="stop after this many shards (default: all of sample-10BT)")
    parser.add_argument("--shard-size", type=int, default=int(1e8),
                        help="tokens per shard (default 100m)")
    args = parser.parse_args()

    shard_size = args.shard_size
    max_shards = args.shards

    data_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_dir)
    os.makedirs(data_cache_dir, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

    nprocs = max(1, os.cpu_count() // 2)
    # not a `with` block on purpose: on the --shards path we exit from inside the
    # loop, and Pool.__exit__ would join a feeder thread that is still blocked
    # reading the stream (see the os._exit note below)
    pool = mp.Pool(nprocs)
    try:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                # room left in the current shard, just append
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # fill the shard to the brim, write it, carry the remainder into the next one
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(shard_path(data_cache_dir, shard_index), all_tokens_np)
                shard_index += 1
                progress_bar.close()
                progress_bar = None

                if max_shards is not None and shard_index >= max_shards:
                    # every shard is already written and closed. the imap feeder
                    # thread is likely still blocked on a read from the dataset
                    # stream, and joining it (pool.terminate / Pool.__exit__)
                    # would hang forever, so leave without unwinding. kill the
                    # workers first or they log BrokenPipeError on the way out.
                    for child in mp.active_children():
                        child.kill()
                    print(f"wrote {shard_index} shards to {data_cache_dir}")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(0)

                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write whatever is left as the final (short) shard
        if token_count != 0:
            if progress_bar is not None:
                progress_bar.close()
            write_datafile(shard_path(data_cache_dir, shard_index), all_tokens_np[:token_count])
            shard_index += 1

        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        raise

    print(f"wrote {shard_index} shards to {data_cache_dir}")


if __name__ == "__main__":
    # macos defaults to the spawn start method, so the pool must be created behind
    # this guard or every child re-runs the script
    main()
