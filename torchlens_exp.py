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
parser.add_argument('--prompt', default='My name is Inigo Montoya you killed my father prepare to die', help='Prompt for text generation.', type=str)
parser.add_argument('--webui', action='store_true', help='If specified, use streamlit-based Web UI instead of command line.')

args = parser.parse_args()

device = (
    "cuda" if torch.cuda.is_available()
    #else "xpu" if ipex.xpu.is_available()
    else "cpu"
)
print(f"Using {device} device")

c2i, i2c = load_token_map(c2i_file=args.c2i, i2c_file=args.i2c)

print(f"Loading model {args.model}")
my_model = torch.load(args.model)
my_model.eval()

print(f"Model vocab_size ({my_model.vocab_size})")

print(f"Using prompt ({args.prompt})")
model_input = torch.tensor(encoder(args.prompt, c2i), device=device).unsqueeze(0)
print(f"Using model_input {model_input}")

# Call torchlens to generate stats on a forward pass of our model
#
# Relevant arguments to log_forward_pass (torchlens/torchlens/user_funcs.py):
#    model:                PyTorch model
#    input_args:           input arguments for model forward pass; as a list if multiple, else as a single tensor.
#    input_kwargs:         keyword arguments for model forward pass
#    layers_to_save:       list of layers to include (described above), defaults to 'all' to include all layers.
#    keep_unsaved_layers:  whether to keep layers without saved activations in the log (i.e., with just metadata). default = True
#    activation_postfunc:  Function to apply to tensors before saving them (e.g., channelwise averaging).
#    mark_input_output_distances: 
#        mark the distance of each layer from the input or output; False by default
#    output_device:        device where saved tensors are to be stored; e.g. 'same'(default), 'cpu', 'cuda'
#    detach_saved_tensors: whether to detach the saved tensors, so they remain attached to the computational graph
#    save_function_args:   whether to save the arguments to each function involved in computing each tensor; default = False
#    save_gradients:       whether to save gradients from any subsequent backward pass
#    vis_opt:              whether, and how, to visualize the network; 
#        'none' (default) for no visualization, 
#        'rolled' to show the graph in rolled-up format (i.e., one node per layer if a recurrent network),
#        'unrolled' to show the graph in unrolled format (i.e., one node per pass through a layer if a recurrent)
#    vis_nesting_depth:    How many levels of nested modules to show; 1 for only top-level modules, 2 for two levels, etc.
#    vis_outpath:          file path to save the graph visualization
#    vis_save_only:        whether to only save the graph visual without immediately showing it
#    vis_fileformat:       the format of the visualization; e.g,. 'pdf' (default), 'jpg', etc.
#    vis_buffer_layers:    whether to visualize the buffer layers
#    vis_direction:        'bottomup' (default), 'topdown', or 'leftright'
#    random_seed:          which random seed to use in case model involves randomness; default = None
#
#
model_history = tl.log_forward_pass(
    my_model, 
    model_input,
    layers_to_save='all', 
    save_function_args=False,
    vis_opt='none',          # Make 'none' for no graph generation, 'unrolled' to generate graph
    vis_save_only=True, 
    vis_direction='leftright',
    print_details=False,
)

# Print resulting stats to a file
logFile = open('layer-stats.txt', 'w')
print(model_history, file = logFile)
logFile.close()
