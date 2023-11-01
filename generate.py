import argparse
import torch
import time
import streamlit as st

import intel_extension_for_pytorch as ipex

from torch.profiler import profile, record_function, ProfilerActivity

# from model import miniGPT
from data import encoder, decoder, load_token_map

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='minigpt.pth', help='Input text for training.', type=str)
parser.add_argument('-l', '--gen_length', default=100, help='Maximum length to generate.', type=int)
parser.add_argument('--c2i', default='c2i.json', help='Token map file (character to index).', type=str)
parser.add_argument('--i2c', default='i2c.json', help='Token map file (index to character).', type=str)
parser.add_argument('--prompt', default=' ', help='Prompt for text generation.', type=str)
parser.add_argument('--webui', action='store_true', help='If specified, use streamlit-based Web UI instead of command line.')

args = parser.parse_args()

print(f"Loading model from {args.model}")
model = torch.load(args.model)

c2i, i2c = load_token_map(c2i_file=args.c2i, i2c_file=args.i2c)
print(f"Loading token map file from {args.c2i} and {args.i2c}")

device = (
    "cuda" if torch.cuda.is_available()
    #else "xpu" if ipex.xpu.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Test with zero input (according to i2c.json, 0 means '\n')
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decoder(model.generate(context, max_new_tokens=args.gen_length)[0].tolist(), i2c))

def gen_response(prompt:str) -> str:
    start = time.time()
    # encode the prompt into a tensor, and reshape it into (1, T)
    # first dimension is the batch, which is expected by the forward method.
    context = torch.tensor(encoder(prompt, c2i), device=device).unsqueeze(0)
    # model.generate() will truncate the prompt if it's too long, no need to worry about this.

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            resp = decoder(model.generate(context, max_new_tokens=args.gen_length)[0].tolist(), i2c)
    end = time.time()
    tokens_generated = min(args.gen_length, len(resp) - len(prompt))
    print(f"{tokens_generated} tokens generated in {end-start:>.3f} seconds, avg {tokens_generated/(end-start):>.3f} tokens/sec.")
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    return resp

def webui():
    st.set_page_config(page_title='Mini GPT for demonstration purpose')
    st.title('Mini GPT')
    with st.form('Generate...'):
        text = st.text_area('Enter prompt:', args.prompt)
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.info(gen_response(text))

if __name__ == "__main__":
    if args.webui:
        webui()
    else:
        print(gen_response(args.prompt))

        
