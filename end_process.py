from itertools import combinations
import networkx as nx
from dynpertnet import *

name_list= ['apo', 'holo', 'holoatp']

output_folder = '/home/agheeraert/TRAJ_AMPK/RESULTS/NEW_ALGO/A2B1'
L_output_atomic = [jn(output_folder, '{}.anpy'.format(name)) for name in name_list]

networks = list(map(nx.read_gpickle, L_output_atomic))

#Making atomic perturbations
for i, j in combinations(range(len(networks)), 2):
    dpn = DynPertNet()
    dpn.create(networks[i], networks[j])
    dpn.save(jn(output_folder, '{0}v{1}.p'.format(name_list[i], name_list[j])))
    dpn_list.append(dpn)

#Converting from atomic to amino_acid


