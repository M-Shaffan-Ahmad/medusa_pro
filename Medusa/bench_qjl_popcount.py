import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _popcount_u32(x):
    x = x.to(tl.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _qjl_popcount_partial_kernel(
    q_bits,
    k_bits,
    partial,
    kv_len: tl.constexpr,
    words: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    node = tl.program_id(0)
    block = tl.program_id(1)
    k_offsets = (block * BLOCK_K) + tl.arange(0, BLOCK_K)
    w_offsets = tl.arange(0, BLOCK_W)

    q = tl.load(q_bits + node * words + w_offsets, mask=w_offsets < words, other=0)
    k = tl.load(
        k_bits + k_offsets[:, None] * words + w_offsets[None, :],
        mask=(k_offsets[:, None] < kv_len) & (w_offsets[None, :] < words),
        other=0,
    )
    matches = _popcount_u32(~(q[None, :] ^ k))
    score_by_word = tl.sum(matches, axis=0)
    score = tl.sum(score_by_word, axis=0)
    tl.store(partial + node * num_blocks + block, score)


def qjl_popcount_scores(q_bits, k_bits, block_k=64):
    if q_bits.dtype != torch.int32 or k_bits.dtype != torch.int32:
        raise TypeError("q_bits and k_bits must be int32 packed words.")
    if q_bits.is_cuda is False or k_bits.is_cuda is False:
        raise ValueError("q_bits and k_bits must be CUDA tensors.")
    nodes, words = q_bits.shape
    kv_len = int(k_bits.shape[0])
    if int(k_bits.shape[1]) != int(words):
        raise ValueError("q_bits and k_bits must have the same packed word count.")
    num_blocks = triton.cdiv(kv_len, block_k)
    partial = torch.empty((nodes, num_blocks), device=q_bits.device, dtype=torch.int32)
    block_w = triton.next_power_of_2(words)
    _qjl_popcount_partial_kernel[(nodes, num_blocks)](
        q_bits,
        k_bits,
        partial,
        kv_len=kv_len,
        words=words,
        num_blocks=num_blocks,
        BLOCK_K=block_k,
        BLOCK_W=block_w,
        num_warps=4,
    )
    return partial.sum(dim=1)


def bench_cuda(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e6 / max(1, repeat), out


def random_int32(shape, device):
    # Random signed words are fine: bitwise XOR and popcount operate on the raw bits.
    return torch.randint(
        -(2**31),
        (2**31) - 1,
        shape,
        device=device,
        dtype=torch.int64,
    ).to(torch.int32)


def main():
    parser = argparse.ArgumentParser(description="Microbenchmark packed 1-bit QJL XNOR-popcount scoring.")
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if args.dim % 32 != 0:
        raise SystemExit("--dim must be divisible by 32 for packed uint32 words.")

    device = "cuda"
    words = args.dim // 32
    q_bits = random_int32((args.nodes, words), device)
    k_bits = random_int32((args.kv_len, words), device)

    q_fp16 = torch.randn(args.nodes, args.dim, device=device, dtype=torch.float16)
    k_fp16 = torch.randn(args.kv_len, args.dim, device=device, dtype=torch.float16)

    qjl_us, qjl_out = bench_cuda(
        lambda: qjl_popcount_scores(q_bits, k_bits, block_k=args.block_k),
        repeat=args.repeat,
    )
    fp16_us, fp16_out = bench_cuda(
        lambda: torch.matmul(q_fp16, k_fp16.t()).amax(dim=1),
        repeat=args.repeat,
    )

    fp16_key_bytes = args.kv_len * args.dim * 2
    qjl_key_bytes = args.kv_len * words * 4
    print("shape", f"nodes={args.nodes}", f"kv_len={args.kv_len}", f"dim={args.dim}", f"words={words}")
    print("qjl_popcount_us", f"{qjl_us:.2f}")
    print("fp16_qk_us", f"{fp16_us:.2f}")
    print("speedup_vs_fp16_qk", f"{fp16_us / max(qjl_us, 1e-6):.2f}x")
    print("fp16_key_mb", f"{fp16_key_bytes / (1024**2):.3f}")
    print("qjl_1bit_key_mb", f"{qjl_key_bytes / (1024**2):.3f}")
    print("key_cache_byte_reduction", f"{fp16_key_bytes / max(qjl_key_bytes, 1):.1f}x")
    print("sample_qjl_score", int(qjl_out[0].item()))
    print("sample_fp16_score", float(fp16_out[0].item()))


if __name__ == "__main__":
    main()
