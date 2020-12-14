import networkx as nx
import pandas as pd 
import seaborn as sns 
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
import multiprocessing
from os import makedirs as mkdir
from maker import *
from itertools import combinations
from os.path import join as jn
from Bio.PDB.Polypeptide import aa1, aa3
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
three2one = dict(zip(aa3, aa1))
t2o = lambda X: three2one[X] if X in three2one else X[0] 


class DynPertNet():
    """Class handling dynamical perturbation networks"""
    def __init__(self):
        pass

    def create(self, net1, net2):
        net1 = self.smart_loader(net1)
        net2 = self.smart_loader(net2)
        id2label = dict(zip(range(len(net1.nodes())), list(net1.nodes())))
        pn_signed_adj = nx.to_numpy_array(net2) - nx.to_numpy_array(net1)
        self.net = nx.from_numpy_array(pn_signed_adj)
        self.net = nx.relabel_nodes(self.net, id2label)

    def smart_loader(self, net):
        if type(net) == AANet:
            return net.net
        elif type(net) == nx.classes.graph.Graph:
            return net
        elif type(net) == str:
            return nx.read_gpickle(net)
        else:
            print('Warning, type of network not detected')
            return net

    def save(self, output):
        """Parameters: path: str
        Saves the network at the given path"""
        nx.write_gpickle(self.net, output)

    def load(self, input):
        """Parameters: path: str
        Loads the network saved at the given path"""
        self.net = nx.read_gpickle(input)
        self.method=None

    def apply_threshold(self, threshold):
        """Parameters: threshold: number
        Makes a copy of the network and applies a threshold to the network"""
        print('Applying threshold {0} on network.'.format(threshold))
        self.current_threshold = threshold
        if threshold != None:
            self.copy = self.net.copy()
            edgelist_to_remove = [(u,v) for u, v in self.net.edges() 
            if abs(self.net.edges()[(u,v)]['weight']) <= threshold]
            self.net.remove_edges_from(edgelist_to_remove)
            self.net.remove_nodes_from(list(nx.isolates(self.net)))   

    def reset(self):
        """ Resets the network to its orginal copy 
        """
        print('Network reset to its original copy')
        self.net = self.copy.copy()
        self.current_threshold = None
        self.method = None

    def get_optimal_threshold(self, method, **kwargs):
        """Parameters: method : str
        Returns the optimal threshold according to different methods.
        The methods are:
        -- tail: Selects the x last percent of edges in the graph 
        (kwargs: tail: number, proportion of edges to keep, default: 0.01)
        -- cluster: Selects the top clusters with max x edges 
        (kwargs: max_edges: int, number of max edges to keep, default: 50)
        -- component: Selects the edges based on a convex component analysis
        (kwargs: eps: number, precision on the optimal threshold, default: 0.1)
        Returns: threshold, number
        """
        if method=='tail':
            return self.tail(**kwargs)
        elif method=='cluster':
            return self.cluster(**kwargs)
        elif method=='component':
            return self.component(**kwargs)
        else:
            return None

    def tail(self, tail=0.01):
        _, __ = plt.subplots(1,1)
        edges_weight = [self.net[u][v]['weight'] for u, v in self.net.edges()]
        df = pd.DataFrame({'weight':edges_weight})
        g = sns.ecdfplot(data=df, x='weight', complementary=True, stat='proportion', ax=__)
        line = np.array(g.lines[0].get_data())
        intersection = line[0][np.argwhere(np.diff(np.sign(line[1]-tail)))]
        plt.close(_)
        return intersection[0][0]

    def cluster(self, max_edges=50):
        weights = np.array([self.net.edges()[(u,v)]['weight'] 
        for (u, v) in self.net.edges()]).reshape(-1, 1)

        birch = Birch(n_clusters=None).fit(weights)
        labels = birch.predict(weights)

        num_labels = np.max(labels)+1

        order = np.argsort(weights[:,0])
        ordered_labels = labels[order]
        ordered_weights = weights[order]

        thresh = []
        num_elements = []
        for label in range(num_labels):
            id_label = ordered_labels == label
            _weights = ordered_weights[id_label]
            X = np.where(ordered_labels == label)
            thresh.append(min(_weights)[0])
            num_elements.append(X[0].shape[0])

        cumsum_elements = np.cumsum(np.sort(np.array(num_elements)))
        df = pd.DataFrame({'Label': range(num_labels), 
                            'Threshold': thresh, 
                            'Number of elements': num_elements}).sort_values(
                                by=['Threshold'], 
                                ascending=False)

        df['Cumulative'] = cumsum_elements
        frontier_row = df.loc[df['Cumulative']<=max_edges].iloc[-1:]
        sugg_thresh = frontier_row['Threshold'].values
        if len(sugg_thresh) == 0:
            return None
        else:
            return sugg_thresh[0]

    def component(self, eps=0.1):
        thresh = 0
        n_compo = []
        _net = self.net.copy()
        while len(_net.edges()) !=0:
            for u, v in _net.edges():
                if _net.edges()[(u, v)]['weight'] <= thresh:
                    _net.remove_edge(u,v)
            _net.remove_nodes_from(list(nx.isolates(_net)))
            n_compo.append(nx.number_connected_components(_net))
            thresh += eps
        n_compo = np.array(n_compo)
        mean = np.mean(n_compo)
        std = np.std(n_compo)
        extra = n_compo <= mean+std     
        return np.where(extra==False)[0][-1]*eps+1

    def apply_optimal_threshold(self, method, **kwargs):
        threshold = self.get_optimal_threshold(method, **kwargs)
        if threshold == None:
            threshold = 0
        threshold = round(threshold, 2)
        self.method = method
        print('Optimal threshold for method {0}: {1}'.format(method.capitalize(), threshold))
        self.apply_threshold(threshold)

    def get_pos_2D(self, pdb_path):
        structure = PDBParser().get_structure('X', pdb_path)[0]
        pos = {}
        for atom in structure.get_atoms():
            if atom.id == 'CA':
                residue = atom.parent
                "these values represents the 2D projection of IGPS in our classical view"
                y = (0.1822020302*atom.coord[0] + 0.6987674421*atom.coord[1] - 0.6917560857*atom.coord[2])
                x = 0.9980297273*atom.coord[0]+ 0.0236149631*atom.coord[1]+ 0.05812914*atom.coord[2]
                pos[t2o(residue.resname)+str(residue.id[1])+':'+residue.parent.id] = (x, y)    
        self.pos_2D = pos

    def load_pos_2D(self, path):
        self.pos_2D = pkl.load(open(path, 'rb'))
    
    def draw(self, ax=None, pdb_path=None, iterations=10):
        """Parameters: 
        pbd_path: str: optional, path of the pdb file to draw on. If structure
        was already loaded in 2D, is not necessary.
        ax: Matplotlib AxesSubplot instance, optional: ax on which to draw the 
        PertNet
        iterations: int: Number of iterations to spring the nodes, default=5
        """
        if ax == None:
            ax = plt.gca()
        
        ax.axis("off")
        
        if self.method:
            ax.set_title("Network at threshold {0} \n (Method {1})".format(
                self.current_threshold, self.method))
        else:
            ax.set_title("Network at threshold {0}".format(self.current_threshold))

        if not self.pos_2D:
            self.pos_2D(pdb_path)
                  
        #Springing nodes
        nodes = self.net.nodes()
        _pos = nx.spring_layout(nodes, pos=self.pos_2D, iterations=5)

        # nx.draw(self.net)   
        #Drawing nodes
        nx.draw_networkx_nodes(self.net, 
                            pos=_pos, 
                            node_size=200, 
                            node_shape='o', 
                            node_color='lightgrey',
                            ax=ax
                            )
        #Handling labels
        labels = {node: node[:-2] if node in nodes else '' 
        for node in self.net.nodes()}
        nx.draw_networkx_labels(self.net, 
                                pos=_pos, 
                                labels=labels, 
                                font_size=10, 
                                font_weight='bold',
                                ax=ax
                            )
        #Handling edges
        colors = [self.net.edges()[(u, v)]['color'] for u, v in self.net.edges()]
        # convert_colors = lambda color: 'b' if color == 'g' else 'l'
        # print(np.apply_along_axis(convert_colors, 0, colors.reshape(-1, 1)))
        # print(colors)
        colors = list(map(lambda x: 'b' if x=='g' else 'r', colors))
        nx.draw_networkx_edges(self.net, 
                                pos=_pos, 
                                width=5, 
                                alpha=1,
                                edge_color=colors,
                                ax=ax
                                )

    def get_pos_3D(self, pdb_path):
        structure = PDBParser().get_structure('X', pdb_path)[0]
        node2CA = {}
        for atom in structure.get_atoms():
            if atom.id == 'CA':
                residue = atom.parent
                coords = ' '.join(map(str, atom.coord))
                node2CA[t2o(residue.resname)+str(residue.id[1])+':'+residue.parent.id] = coords
        
        self.pos_3D = node2CA

    def to_vmd(self, output, pdb_path=None, norm=1.5, same=False):
        """Outputs a .tcl script to use in vmd to load the network on the
        protein.
        Parameters: 
        output: str: path of the output
        pbd_path: str: optional, path of the pdb file to draw on. If structure
        was already loaded in 3D, is not necessary.
        norm: number, optional: normalization factor 
        same: bool, optional: if True and there's not a same attribute,
        sets a common normalization factor for many graphs, if True and already
         a same attribute, uses the common
        normalization factor"""

        if not hasattr(self, 'pos_3D'):
            self.get_pos_3D(pdb_path)

        output = open(output, 'w')
        output.write('draw delete all \n')
        if not same:
            div = max(nx.get_edge_attributes(self.net, 'weight').values())/norm  
        elif not self.same:
            self.div = max(nx.get_edge_attributes(self.net, 'weight').values())/norm
            div = self.div
        else:
            div = self.div 

        #Drawing edges

        color = lambda x: 'blue' if x=='g' else 'red'
        previous = None

        for u, v in self.net.edges():
            c = color(self.net.get_edge_data(u, v)['color'])
            if previous != c:
                output.write('draw color {0} \n'.format(c))
                previous = c
            radius = self.net.get_edge_data(u, v)['weight']/div
            output.write('draw cylinder {'+self.pos_3D[u]+' } { '+self.pos_3D[v]+' } radius '+str(radius)+' \n')
        
        #Drawing nodes
        output.write('draw color silver \n')
        for u in self.net.nodes():
            output.write('draw sphere { '+self.pos_3D[u]+' } radius '+str(norm)+' \n')
        output.close()
    
    def line_draw(self, ax=None, quantity='weight', title=None):

        if ax == None:
            ax = plt.gca()

        if title == None:
            title = str(quantity).capitalize()

        if quantity == 'absweight':
            adjacency = nx.to_numpy_matrix(self.net)
            q = np.sum(adjacency, axis=-1)/2

        else:
            if quantity != 'weight': 
                print("""Quantity to plot in line draw not recognized, 
                computing weights instead""")
            adjacency = nx.to_numpy_matrix(self.net)
            colors = nx.get_edge_attributes(self.net, 'color')
            sign = {edge: 2*(colors[edge] == 'r') - 1 for edge in colors}
            nx.set_edge_attributes(self.net, sign, 'sign')
            signs = nx.to_numpy_matrix(self.net, weight='sign')
            adjacency = np.multiply(adjacency,signs)
            q = np.sum(adjacency, axis=-1)/2

        ids = [int(node[1:-2]) for node in self.net.nodes()]
        ax.set_title(title)
        ax.plot(ids, q, color='k')
        ax.set_xlabel('Residue number')
        return ids, q
            
def create_dpn(traj1, traj2, topo=None, topo1=None, topo2=None, selection='all', cutoff=5, out1=None, out2=None):
    if topo:
        topo1, topo2 = topo, topo
    aanet1 = create_aanet(traj1, topo=topo1, selection=selection, cutoff=cutoff)
    if out1:
        aanet1.save(out1)
    aanet2 = create_aanet(traj2, topo=topo2, selection=selection, cutoff=cutoff)
    if out2:
        aanet2.save(out2)
    dpn = DynPertNet()
    dpn.create(aanet1, aanet2)
    return dpn
 
def create_dpn_parallel(traj_list, topo_list, selection='all', cutoff=5, output_folder=None, name_list=None):
    n_cpu = multiprocessing.cpu_count()
    n_trajs = len(traj_list)
    if type(topo_list) != list:
        topo_list = [topo_list]*n_trajs
    if output_folder == None or name_list==None:
        output_list = [None]*n_trajs
    else:
        output_list = [jn(output_folder, '{0}.p'.format(name)) for name in name_list]

    selection = [selection]*n_trajs
    cutoff = [cutoff]*n_trajs

    pool = multiprocessing.Pool(processes=min(n_cpu, n_trajs))
    networks = pool.starmap(create_aan_parallel, zip(traj_list, topo_list, selection, cutoff, output_list))
    dpn = []
    for i, j in combinations(range(len(networks)), 2):
        dpn = DynPertNet()
        dpn.create(networks[i], networks[j])
        if output_folder != None:
            dpn.save(jn(output_folder, '{0}v{1}.p'.format(name_list[i], name_list[j])))
     

def create_aan_parallel(traj, topo, selection, cutoff, output):
    aanet = AANet()
    aanet.create(traj, topo, selection, cutoff)
    if output != None:
        aanet.save(output)
    return aanet.net

def create_dpn_from_pickle(inp1, inp2):
    dpn = DynPertNet()
    dpn.create(inp1, inp2)
    return dpn

def load_dpn(path):
    dpn = DynPertNet()
    dpn.load(path)
    return dpn

def create_list(traj1, traj2, selectionList, topo=None, topo1=None, topo2=None, selection='all', cutoff=5, output_atomic=None, output_aanet=None, output=None):
    if topo:
        topo1, topo2 = topo, topo
    aanet1_list = create_aanet_list(traj1, selectionList=selectionList, topo=topo1, selection=selection, cutoff=cutoff, output_atomic=output_atomic[0], output_list=output_aanet[0])
    aanet2_list = create_aanet_list(traj2, selectionList=selectionList, topo=topo2, selection=selection, cutoff=cutoff, output_atomic=output_atomic[1], output_list=output_aanet[1])
    dpn_list, i = [], 0
    for aanet1, aanet2 in zip(aanet1_list, aanet2_list):
        dpn = DynPertNet()
        dpn.create(aanet1, aanet2)
        dpn_list.append(dpn)
        try:
            dpn.save(output[i])
        except Exception as e: print(e)
        i+=1

    return dpn_list

def create_default(traj1, traj2, topo, output_folder, name1, name2):
    selectionList = ['all', 'not hydrogen', 'backbone || name H HA', 'backbone', 'sidechain', 'sidechain && not hydrogen', ['all', 'name H N']]
    outs = ['allH', 'all', 'backboneH', 'backbone', 'sidechainH', 'sidechain', 'amide_proton']
    mkdir(jn(output_folder, 'atomic'), exist_ok=True)
    output_atomic = [jn(output_folder, 'atomic', '{0}.p'.format(name)) for name in [name1, name2]]
    mkdir(jn(output_folder, 'aa_networks'), exist_ok=True)
    output_aanet = [[jn(output_folder, 'aa_networks', '{0}_{1}.p'.format(selection, name1)) for selection in outs],
                   [jn(output_folder, 'aa_networks', '{0}_{1}.p'.format(selection, name2)) for selection in outs]]
    output = [jn(output_folder, '{0}.p'.format(selection)) for selection in outs]

    dpn_list = create_list(traj1, traj2, selectionList, topo=topo, selection='all', cutoff=5, output_atomic=output_atomic, output_aanet=output_aanet, output=output)

    return dpn_list
    



if __name__ == '__main__':
    DIR = '/home/aria/landslide/MDRUNS/IVAN_IGPS'
    output_folder = '/home/aria/landslide/RESULTS/GUIDELINES/NEW_ALGORITHM/TEST'
    name1 = 'apo'
    name2 = 'holo'
    traj1=[jn(DIR, 'prot_apo_sim{0}_s10.dcd'.format(i)) for i in range(1,5)]
    traj2=[jn(DIR, 'prot_prfar_sim{0}_s10.dcd'.format(i)) for i in range(1,5)]
    topo = jn(DIR, 'prot.prmtop')
    create_default(traj1, traj2, topo, output_folder, name1, name2)



    #OLDER TESTS

    # DIR = '/home/aria/landslide/MDRUNS/YEAST/all_trajs'

    # # dpn = create_dpn(traj1=jn(DIR, 'model1_apo.dcd'),
    # #                       traj2=jn(DIR, 'model1_holo_protein.dcd'),
    # #                       topo='/home/aria/landslide/FRAMES/YEAST/ALLH/APO/frame_1.pdb',
    # #                       selection="not hydrogen",
    # #                       out1='data/apo1.p',
    # #                       out2='data/holo1.p')
    # ###
    # dpn = create_dpn_from_pickle("data/apo1.p", "data/holo1.p")
    # #dpn.save("data/dpn1.p")
    # dpn = load_dpn("data/dpn1.p")
    # dpn.load_pos_2D('data/yeast_IGPS_pos.p') #Precomputed 2D projection
    # dpn.get_pos_3D('data/frame_1.pdb') #Computing directly on the frame

    # method_list = ['tail', 'cluster', 'component']

    # fig, axs = plt.subplots(2, 2, figsize=[15, 10])
    # plt.subplots_adjust(wspace=0.05, hspace=0.1) #Helps reducing margins
    
    # #Drawing first a network at threshold 6
    # old_threshold = 6
    # dpn.apply_threshold(old_threshold)
    # dpn.draw(ax=axs[0, 0])
    # dpn.reset()

    # #Then drawing networks at optimum thresholds
    # for method, ax in zip(method_list, axs.reshape(-1)[1:]):
    #     dpn.apply_optimal_threshold(method)
    #     dpn.draw(ax=ax)
    #     dpn.reset()
    # #bbox_inches=tight allows to remove white space around the networks
    # plt.savefig(jn('data', 'plot_2d_new.png'), bbox_inches='tight')

"""
if __name__ == '__main__':
    #Loading the dynamical perturbation network (0.p)
    dpn = DynPertNet()
    dpn.load('data/all_full.p')
    #Loading positions for 2D and 3D plots
    dpn.load_pos_2D('data/yeast_IGPS_pos.p') #Precomputed 2D projection
    dpn.get_pos_3D('data/frame_1.pdb') #Computing directly on the frame

    #Here I present Some simple examples

    # Apply a given threshold (here 6)
    dpn.apply_threshold(6)
    # Drawing the network at this threshold in 2D
    fig, ax = plt.subplots(1, 1)
    dpn.draw(ax)
    plt.savefig('data/test1.png', bbox_inches='tight')
    # Creating VMD script to load the network on the protein
    dpn.to_vmd('data/test1.tcl')
    # Resetting the network to its original form (without threshold)
    dpn.reset() #Don't forget this step it can lead to some errors

    # Apply a optimal threshold
    dpn.apply_optimal_threshold('tail', tail=0.005) #By default tail is 10%. Here I put 5%
    # Drawing the network at this threshold in 2D
    fig, ax = plt.subplots(1, 1)
    dpn.draw(ax)
    plt.savefig('data/test2.png', bbox_inches='tight')
    # Creating VMD script to load the network on the protein
    dpn.to_vmd('data/test2.tcl')
    # Resetting the network to its original form (without threshold)
    dpn.reset() #Don't forget this step it can lead to some errors

    "Here I use 4 differents methods in a row"
    method_list = ['tail', 'cluster', 'component']

    fig, axs = plt.subplots(2, 2, figsize=[15, 10])
    plt.subplots_adjust(wspace=0.05, hspace=0.1) #Helps reducing margins
    #Drawing first a network at threshold 6
    old_threshold = 6
    dpn.apply_threshold(old_threshold)
    dpn.draw(ax=axs[0, 0])
    dpn.to_vmd(jn('data', 'all_6.tcl'))
    dpn.reset()

    #Then drawing networks at optimum thresholds
    for method, ax in zip(method_list, axs.reshape(-1)[1:]):
        dpn.apply_optimal_threshold(method)
        dpn.draw(ax=ax)
        dpn.to_vmd(jn('data', 'all_{0}.tcl'.format(method)))
        dpn.reset()
    #bbox_inches=tight allows to remove white space around the networks
    plt.savefig(jn('data', 'plot_2d.png'), bbox_inches='tight')

    # Here we use 4 differents methods in a row
    method_list = ['tail', 'cluster', 'component']

    fig, axs = plt.subplots(2, 2, figsize=[15, 10])
    plt.subplots_adjust(wspace=0.05, hspace=0.1) #Helps reducing margins
    #Drawing first a network at threshold 6
    old_threshold = 6
    dpn.apply_threshold(old_threshold)
    dpn.draw(ax=axs[0, 0])
    dpn.to_vmd(jn('data', 'all_6.tcl'))
    dpn.reset()

    #Then drawing networks at optimum thresholds
    for method, ax in zip(method_list, axs.reshape(-1)[1:]):
        dpn.apply_optimal_threshold(method)
        dpn.draw(ax=ax)
        dpn.to_vmd(jn('data', 'all_{0}.tcl'.format(method)))
        dpn.reset()
    #bbox_inches=tight allows to remove white space around the networks
    plt.savefig(jn('data', 'plot_2d.png'), bbox_inches='tight')
    plt.close()

    #Drawing line plot of weights and comparing time
    fig, axs = plt.subplots(2, 1)
    ids, q1 = dpn.line_draw(ax=axs[0], quantity='weight', title='Weight of each node in the perturbation network')
    ids, q2 = dpn.line_draw(ax=axs[1], quantity='absweight', title='Absolute weight of each node in the perturbation network')
    plt.tight_layout()
    plt.savefig('data/line_plots.png')
    print(ids, q1, q2) """




        
