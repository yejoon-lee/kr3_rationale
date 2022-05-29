import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help='path to load attentions and logits')
parser.add_argument("save_path", help='path to save the rationales')
parser.add_argument("--strategy", help='Strategy for discretization, used in Jain et al., 2020', default='Top-k')
parser.add_argument("--ratio", help='ratio of length of rationale to whole input.', default=0.2)
args = parser.parse_args()

print(args.name)
print(args.save_path)
print(args.strategy)
print(args.ratio)