import argparse
import torch
from torchview import draw_graph

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

model = torch.load('minigpt-cpu.pth')
model.eval()

dummy_input = torch.tensor(encoder("dummy prompt", c2i), device=device).unsqueeze(0)

batch_size = 32
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_data=dummy_input, device='meta')
model_graph.visual_graph