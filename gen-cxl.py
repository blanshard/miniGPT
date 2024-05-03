import argparse
import torch
import time
import ctypes

# from model import miniGPT
from data import encoder, decoder, load_token_map
#import intel_extension_for_pytorch as ipex

# python memutils module
import memutils


def small_test(verbose, printDetails=False):
    if verbose:    print("small test: x = torch.empty(1000, 1000)")
    x = torch.empty(1000, 1000)

    if verbose:    print("small test: calling memutils.log_stats(** After x)")
    memutils.log_stats("** After x", clearStats=True, printDetails=printDetails)

    if verbose:    print("small test: y = torch.empty(1000, 1000)")
    y = torch.empty(1000, 1000)

    if verbose:    print("small test: z = torch.empty(500, 250)")
    z = torch.empty(500, 250)

    if verbose:    print("small test: a = torch.empty(2000, 100)")
    a = torch.empty(2000, 100)
    
    if verbose:    print("small test: calling memutils.log_stats(** After y, z and a)")
    memutils.log_stats("** After y, z and a", clearStats=True, printDetails=printDetails)
    
    if verbose:    print("small test: returning, all the tensors will be freed")

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='minigpt.pth', help='Input text for training.', type=str)
    parser.add_argument('-l', '--gen_length', default=100, help='Maximum length to generate.', type=int)
    parser.add_argument('--c2i', default='c2i.json', help='Token map file (character to index).', type=str)
    parser.add_argument('--i2c', default='i2c.json', help='Token map file (index to character).', type=str)
    parser.add_argument('--prompt', default='the quick brown fox jumps over the lazy dog', help='Prompt for text generation.', type=str)
    parser.add_argument('--webui', action='store_true', help='If specified, use streamlit-based Web UI instead of command line.')
    parser.add_argument('--notverbose', action='store_true', help='If specified, suppress progress prints.', default=False)
    parser.add_argument('--statsdetail', action='store_true', help='If specified, print allocation details', default=False)
    parser.add_argument('--umf', action='store_true', help='Use UMF shared memory allocator; default = false', default=False)
    parser.add_argument('--cxl', action='store_true', help='Use CXL; default=false', default=False)
    parser.add_argument('--test', action='store_true', help='Just do small test; default=false', default=False)
    
    args = parser.parse_args()
    print("args:", args)
    
    # Set print verbosity
    verbose = True
    if args.notverbose:
        verbose = False
    
    if verbose:    print("Setting up memory utils")
    memutils.set_period(0)
    logDetails = args.statsdetail
    memutils.use_umf(args.umf)
    memutils.use_cxl(args.cxl)
    statsInCsv = False

    device = (
        "cuda" if torch.cuda.is_available()
        #else "xpu" if ipex.xpu.is_available()
        else "cpu"
    )
    if verbose:    print(f"Using {device} device")
    
    c2i, i2c = load_token_map(c2i_file=args.c2i, i2c_file=args.i2c)
    print(f"Loading token map file from {args.c2i} and {args.i2c}")
    
    print(f"Loading model from {args.model}")
    model = torch.load(args.model)
    memutils.log_stats("After loading model", clearStats=True, printDetails=logDetails, csv=statsInCsv)

    # Generate response
    
    start = time.time()
    # encode the prompt into a tensor, and reshape it into (1, T)
    # first dimension is the batch, which is expected by the forward method.
    context = torch.tensor(encoder(args.prompt, c2i), device=device, pin_memory=True).unsqueeze(0)
    memutils.log_stats(prefix="After creating context", printDetails=logDetails, clearStats=True, csv=True)

    response = decoder(model.generate(context, max_new_tokens=args.gen_length)[0].tolist(), i2c)
    end = time.time()
    tokens_generated = min(args.gen_length, len(response) - len(args.prompt))
    memutils.log_stats(prefix="After generate and decode", printDetails=logDetails, clearStats=True, csv=True)

    if verbose:
        print(f"{tokens_generated} tokens generated in {end-start:>.3f} seconds, avg {tokens_generated/(end-start):>.3f} tokens/sec.")
        print("Response:")
        print(response)
    
    return


if __name__ == "__main__":
    main()