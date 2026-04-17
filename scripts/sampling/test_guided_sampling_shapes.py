"""Test script to verify guided sampling works with correct input shapes."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from evo.tokenization import Vocab
from evo.oracles import get_oracle
from cosine.models.modules.ctmc_module import CTMCModule
from cosine.models.nets.ctmc import NeuralCTMC, NeuralCTMCGenerator

def test_guided_sampling():
    """Test that guided sampling works with sequences including special tokens."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load model
    model_path = Path("/scratch/users/stephen.lu/projects/protevo/logs/train/runs/2026-01-06_18-32-49/checkpoints/epoch_001.ckpt")
    print(f"Loading model from: {model_path}")
    module = CTMCModule.load_from_checkpoint(str(model_path), map_location=device, strict=False)
    net: NeuralCTMC = module.net
    vocab: Vocab = net.vocab
    net = net.eval().to(device)
    generator = NeuralCTMCGenerator(neural_ctmc=net)

    print(f"vocab.prepend_bos: {vocab.prepend_bos}")
    print(f"vocab.append_eos: {vocab.append_eos}")

    # Load oracle
    print("\nLoading SARSCoV1 oracle...")
    oracle = get_oracle(
        oracle_name="SARSCoV1",
        device=device,
        enable_mc_dropout=True,
        mc_samples=10,
        cache_size=10000
    )

    # Test sequence (119 amino acids from Rodriguez dataset)
    test_seq = "CAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTCAAGCCTGGAGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTCAGTGACTACTACATGAGCTGGATCCGCCAGGCT"
    print(f"\n{'='*80}")
    print("TEST SEQUENCE")
    print(f"{'='*80}")
    print(f"Length (amino acids only): {len(test_seq)}")
    print(f"Sequence: {test_seq[:60]}...")

    # Encode WITH special tokens (this is the correct way)
    parent_encoded = vocab.encode_single_sequence(test_seq)
    print(f"\n{'='*80}")
    print("ENCODING WITH SPECIAL TOKENS")
    print(f"{'='*80}")
    print(f"Encoded length (with BOS/EOS): {len(parent_encoded)}")
    print(f"Expected length: {len(test_seq)} + {int(vocab.prepend_bos)} (BOS) + {int(vocab.append_eos)} (EOS) = {len(test_seq) + int(vocab.prepend_bos) + int(vocab.append_eos)}")
    print(f"First 5 tokens: {parent_encoded[:5]}")
    print(f"Last 5 tokens: {parent_encoded[-5:]}")
    print(f"First 5 decoded: {[vocab.token(idx) for idx in parent_encoded[:5]]}")
    print(f"Last 5 decoded: {[vocab.token(idx) for idx in parent_encoded[-5:]]}")

    # Create batch
    batch_size = 3
    x = torch.from_numpy(parent_encoded).unsqueeze(0).repeat(batch_size, 1).to(device)

    # x_sizes should include special tokens
    root_length_with_special = len(test_seq) + int(vocab.prepend_bos) + int(vocab.append_eos)
    x_sizes = torch.tensor([[root_length_with_special]], device=device)

    print(f"\n{'='*80}")
    print("INPUT SHAPES")
    print(f"{'='*80}")
    print(f"x shape: {x.shape} (expected: ({batch_size}, {len(parent_encoded)}))")
    print(f"x_sizes shape: {x_sizes.shape} (expected: (1, 1))")
    print(f"x_sizes value: {x_sizes.item()} (expected: {root_length_with_special})")

    # Test expanding x_sizes for batch
    x_sizes_batch = x_sizes.repeat(x.size(0), 1)
    print(f"x_sizes_batch shape: {x_sizes_batch.shape} (expected: ({batch_size}, 1))")
    print(f"x_sizes_batch values: {x_sizes_batch.squeeze().tolist()}")

    # Branch lengths
    t = torch.tensor([0.05] * batch_size, device=device)

    print(f"\n{'='*80}")
    print("TEST 1: UNGUIDED SAMPLING")
    print(f"{'='*80}")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_unguided = generator.generate_with_gillespie(
            x=x,
            t=t,
            x_sizes=x_sizes_batch,
            max_decode_steps=100,
            verbose=False
        )

    print(f"Generated shape: {y_unguided.shape} (expected: ({batch_size}, {len(parent_encoded)}))")

    # Decode and check
    for i in range(batch_size):
        seq_str = "".join([vocab.token(idx.item()) for idx in y_unguided[i]
                          if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])
        print(f"Sample {i}: length={len(seq_str)}, seq={seq_str[:40]}...")

    print(f"\n{'='*80}")
    print("TEST 2: GUIDED SAMPLING (Taylor Approximation)")
    print(f"{'='*80}")

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_guided = generator.generate_with_gillespie(
                x=x,
                t=t,
                x_sizes=x_sizes_batch,
                oracle=oracle,
                guidance_strength=1.0,
                use_taylor_approx=True,
                use_guidance=True,
                max_decode_steps=100,
                verbose=True,
                oracle_chunk_size=10
            )

        print(f"Generated shape: {y_guided.shape} (expected: ({batch_size}, {len(parent_encoded)}))")

        # Decode and check
        for i in range(batch_size):
            seq_str = "".join([vocab.token(idx.item()) for idx in y_guided[i]
                              if vocab.token(idx.item()) in set("ARNDCQEGHILKMFPSTWYV")])
            print(f"Sample {i}: length={len(seq_str)}, seq={seq_str[:40]}...")

        print(f"\n{'='*80}")
        print("✓ ALL TESTS PASSED!")
        print(f"{'='*80}")
        print("Guided sampling works correctly with sequences including special tokens.")
        print(f"Input shape: ({batch_size}, {len(parent_encoded)}) with BOS/EOS")
        print(f"x_sizes: {root_length_with_special} (includes special tokens)")

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print("✗ TEST FAILED!")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_guided_sampling()
    sys.exit(0 if success else 1)
