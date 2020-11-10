import networkx as nx
import pandas as pd 
import seaborn as sns 
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import Birch
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
    def __init__(self, path):
        """Parameters: path: str
        Loads the network saved at the given path"""
        self.net = nx.read_gpickle(path)
        self.method=None

    def apply_threshold(self, threshold):
        """Parameters: threshold: number
        Makes a copy of the network and applies a threshold to the network"""
        print('Applying threshold {0} on network.'.format(threshold))
        self.current_threshold = threshold
        if threshold != None:
            self.copy = self.net.copy()
            edgelist_to_remove = [(u,v) for u, v in self.net.edges() 
            if self.net.edges()[(u,v)]['weight'] <= threshold]
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
        threshold = round(self.get_optimal_threshold(method, **kwargs), 2)
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

    
if __name__ == '__main__':
    #Loading the dynamical perturbation network (0.p)
    dpn = DynPertNet('data/all_full.p')
    #Loading positions for 2D and 3D plots
    dpn.load_pos_2D('data/yeast_IGPS_pos.p') #Precomputed 2D projection
    dpn.get_pos_3D('data/frame_1.pdb') #Computing directly on the frame

    """Here I present Some simple examples"""

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

    """Here I use 4 differents methods in a row"""
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





        
