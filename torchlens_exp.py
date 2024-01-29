import argparse
import torch
import torchlens as tl

import intel_extension_for_pytorch as ipex
from data import encoder, decoder, load_token_map

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='minigpt.pth', help='Input text for training.', type=str)
parser.add_argument('-l', '--gen_length', default=100, help='Maximum length to generate.', type=int)
parser.add_argument('--c2i', default='c2i.json', help='Token map file (character to index).', type=str)
parser.add_argument('--i2c', default='i2c.json', help='Token map file (index to character).', type=str)
parser.add_argument('--prompt', default=' ', help='Prompt for text generation.', type=str)
parser.add_argument('--webui', action='store_true', help='If specified, use streamlit-based Web UI instead of command line.')

args = parser.parse_args()

device = (
    "cuda" if torch.cuda.is_available()
    #else "xpu" if ipex.xpu.is_available()
    else "cpu"
)
print(f"Using {device} device")

c2i, i2c = load_token_map(c2i_file=args.c2i, i2c_file=args.i2c)

my_model = torch.load('minigpt-cpu.pth')
my_model.eval()

dummy_input = torch.tensor(encoder("dummy input", c2i), device=device).unsqueeze(0)

print(dummy_input)

#model_history = tl.log_forward_pass(my_model, dummy_input, layers_to_save='all', vis_opt='unrolled')
model_history = tl.log_forward_pass(my_model, dummy_input, layers_to_save='all', vis_opt='none')

logFile = open('log.txt', 'w')
print(model_history, file = logFile)
logFile.close()
