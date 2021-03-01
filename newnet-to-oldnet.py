import networkx as nx
import argparse

parser = argparse.ArgumentParser(description='Convert new DPN to old format')
parser.add_argument('f',  type=str, nargs='+',
                     help='files to convert')
args = parser.parse_args()

for f in args.f:
    net = nx.read_gpickle(f)
    copy = net.copy()
    for u, v in net.edges():
        copy[u][v]['weight'] = abs(net[u][v]['weight'])
        sign2color = lambda X: 'r' if X>=0 else 'g'
        copy[u][v]['color'] = sign2color(net[u][v]['weight'])
    if list(net.nodes())[0][1:-2]=='0':
        mapping = {elt: elt[0]+str(int(elt[1:])+1)+':X'  for elt in net.nodes()}
    else:
        mapping = {elt: elt+':X'  for elt in net.nodes()}
    copy = copy.relabel_nodes(mapping)
    nx.write_gpickle(copy, f)