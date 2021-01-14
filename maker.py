from multiprocessing import process
import networkx as nx 
import numpy as np
from scipy.spatial import cKDTree
import mdtraj as md
import pickle as pkl
import multiprocessing
from os.path import join as jn
from Bio.PDB.Polypeptide import aa1, aa3
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
three2one = dict(zip(aa3, aa1))
t2o = lambda X: three2one[X] if X in three2one else X[0]
label =  lambda X: t2o(X.name)+str(X.index)
from tqdm import tqdm
from scipy.sparse import csr_matrix

class AANet():
    """Class to create an AANetwork from a trajectory"""
    def __init__(self):
        pass

    def load(self, input):
        nx.read_gpickle(input)

    def create(self, traj, topo=None, selection='all', cutoff=5):
        #Loading trajectory
        t = md.load(traj, top=topo)
        #Slicing atoms of interest
        if selection != 'all':
            t = t.atom_slice(t.topology.select(selection))
        
        #Creating our topological matrix
        n_atoms, n_residues = t.topology.n_atoms, t.topology.n_residues
        labels = list(map(label, t.topology.residues))
        self.id2label = dict(zip(list(range(n_residues)), labels))

        top_mat = np.zeros((n_atoms, n_residues))        
        for atom in t.topology.atoms:
            top_mat[atom.index, atom.residue.index] = 1
        top_mat = csr_matrix(top_mat)

        #Getting the atomic contacts
        coords = t.xyz
        self.contacts = []
        for frame in tqdm(range(t.n_frames)):
            #Here we're using the cPython KDTree algorithm to get the neighbors
            tree = cKDTree(coords[frame])
            pairs = tree.query_pairs(r=cutoff/10.) #Cutoff is in Angstrom but mdtraj uses nm
            #Creating sparse CSR matrix
            data = np.ones(len(pairs))
            pairs = np.array(list(pairs))
            atoms = csr_matrix((data, (pairs[:,0], pairs[:,1])), shape=[n_atoms, n_atoms])
            #R=T^t.A.T where R is residue contact matrix, A si atomic contact matrix and T our topological matrix
            self.contacts.append(csr_matrix(np.dot(top_mat.transpose(),atoms.dot(top_mat))))
        
        #Computing average from list of csr matrices
        average = np.zeros((n_residues, n_residues))
        for mat in self.contacts:
            average[mat.nonzero()] += mat.data/t.n_frames
        self.average = average
        self.net = nx.from_numpy_array(self.average)
        #Labeling the network
        self.net = nx.relabel_nodes(self.net, self.id2label, copy=False)
    
    def create_parallel(self, traj, topo=None, selection='all', cutoff=5, n_procs=1):
        t = md.load(traj, top=topo)
        if selection != 'all':
            t = t.atom_slice(t.topology.select(selection))
        #Creating our topological matrix
        n_atoms, n_residues = t.topology.n_atoms, t.topology.n_residues
        labels = list(map(label, t.topology.residues))
        self.id2label = dict(zip(list(range(n_residues)), labels))

        top_mat = np.zeros((n_atoms, n_residues))        
        for atom in t.topology.atoms:
            top_mat[atom.index, atom.residue.index] = 1
        top_mat = csr_matrix(top_mat)

        #Getting the atomic contacts
        coords = t.xyz
        #Chunk the contacts
        coords = np.array_split(coords, n_procs)

        def create_chunk(coords):
            contacts = 0
            n_frames = coords.shape[0]
            for frame in tqdm(n_frames):
                #Here we're using the cPython KDTree algorithm to get the neighbors
                tree = cKDTree(coords[frame])
                pairs = tree.query_pairs(r=cutoff/10.) #Cutoff is in Angstrom but mdtraj uses nm
                #Creating sparse CSR matrix
                data = np.ones(len(pairs))
                pairs = np.array(list(pairs))
                atoms = csr_matrix((data, (pairs[:,0], pairs[:,1])), shape=[n_atoms, n_atoms])
                #R=T^t.A.T where R is residue contact matrix, A si atomic contact matrix and T our topological matrix
                contacts.append(csr_matrix(np.dot(top_mat.transpose(), atoms.dot(top_mat))))            
            #Computing average from list of csr matrices
            average = np.zeros((n_residues, n_residues))
            for mat in self.contacts:
                average[mat.nonzero()] += mat.data
            return average
        
        pool = multiprocessing.Pool(processes=n_procs)
        chunk_contacts = pool.map(create_chunk, coords)
        self.average = sum(chunk_contacts)/t.n_frames
        self.net = nx.from_numpy_array(self.average)
        #Labeling the network
        self.net = nx.relabel_nodes(self.net, self.id2label, copy=False)


    def create_atomic(self, traj,  baseSelection, topo=None, cutoff=5):
        """Function creating the atomic contact network with a desired base selection.
        Parameters: traj: str or list of str: path trajectories to load
        topo: str: path of topology to use
        baseSelection: str: base selection on which to compute the atomic network. To save computation time, this should be the 
        smallest selection that includes all the selections in the list.
        """

        self.t = md.load(traj, top=topo)
        #Slicing atoms of interest
        if baseSelection != 'all':
            self.t = self.t.atom_slice(self.t.topology.select(baseSelection))

        self.n_atoms, self.n_residues = self.t.topology.n_atoms, self.t.topology.n_residues
        labels = list(map(label, self.t.topology.residues))
        self.id2label = dict(zip(list(range(self.n_residues)), labels))
        coords = self.t.xyz
        atomicContacts = []
        for frame in tqdm(range(self.t.n_frames)):
            #Here we're using the cPython KDTree algorithm to get the neighbors
            tree = cKDTree(coords[frame])
            pairs = tree.query_pairs(r=cutoff/10.) #Cutoff is in Angstrom but mdtraj uses nm
            #Creating sparse CSR matrix
            data = np.ones(len(pairs))
            pairs = np.array(list(pairs))
            atomicContacts.append(csr_matrix((data, (pairs[:,0], pairs[:,1])), shape=[self.n_atoms, self.n_atoms]))            
        
        #Computing average atomic network from list of csr matrices
        self.atomic_avg = csr_matrix((self.n_atoms, self.n_atoms))
        for mat in atomicContacts:
            self.atomic_avg += mat
        self.atomic_avg /= self.t.n_frames

    def save_atomic(self, output):
        """Saves atomic network to the desired output
        Parameters: output: str, path where to output the file"""
        nx.write_gpickle(self.atomic_avg, output)

    def load_atomic(self, input):
        """Loads atomic network
        Parameters: output: str, path where to input the file"""
        self.atomic_avg = nx.read_gpickle(input)
    
    
    def create_list(self, selectionList):
        """Function to apply to a atomic contact network and then builds from it a list of amino acid network
        with desired selections.
        Parameters: traj: str or list of str: path trajectories to load
        selectionList: list of str or list of tuple of str: Each element can be a tuple in order to create asymmetric selections
        topo: str: path of topology to use
        baseSelection: str: base selection on which to compute the atomic network. To save computation time, this should be the 
        smallest selection that includes all the selections in the list.
        """

        def create_top(selection):
            selection = selection.replace("not hydrogen", "!(name =~'H.*')")
            indexes = self.t.topology.select(selection)
            top_mat = np.zeros((self.n_atoms, self.n_residues))        
            for atom in self.t.topology.atoms:
                if atom.index in indexes:
                    top_mat[atom.index, atom.residue.index] = 1
            top_mat = csr_matrix(top_mat)
            return top_mat

        #Getting the atomic contacts
        networks = []

        for selection in selectionList:
            if len(selection)==2:
                #Means we've got an asymmetric selection to handle
                T1=create_top(selection[0]).transpose()
                T2=create_top(selection[1])
            else:
                T2=create_top(selection)
                T1=T2.transpose()
            mat = T1.dot(self.atomic_avg.dot(T2))
            net = nx.from_scipy_sparse_matrix(mat)
            networks.append(nx.relabel_nodes(net, self.id2label, copy=False))
        return networks
            

    def save(self, output):
        nx.write_gpickle(self.net, output)

    def apply_threshold(self, threshold):
        copy = self.net.copy()
        edgelist = []
        for u, v in self.net.edges():
            if self.net[u][v]['weight'] >= threshold:
                edgelist.append((u,v))
        copy.remove_edges_from(edgelist)
        copy.remove_nodes_from(list(nx.isolates(copy)))
        return copy

def create_aanet(traj, topo=None, selection='all', cutoff=5):
    selection = selection.replace("not hydrogen", "!(name =~'H.*')")
    aanet = AANet()
    aanet.create(traj, topo=topo, selection=selection, cutoff=cutoff)
    return aanet

def load_aanet(input):
    aanet = AANet()
    aanet.load(input)
    return aanet

def create_aanet_multiselection(traj, selectionList, topo=None, selection='all', cutoff=5, output_atomic=None, output_list = None):
    selection = selection.replace("not hydrogen", "!(name =~'H.*')")
    aanet = AANet()
    aanet.create_atomic(traj, baseSelection=selection, topo=topo, cutoff=cutoff)
    if output_atomic:
        aanet.save_atomic(output_atomic)
    networks = aanet.create_list(selectionList)
    if output_list:
        for net, out in zip(networks, output_list): 
            nx.write_gpickle(net, out)
    return networks


if __name__ == '__main__':

######### TESTING IF CREATE_LIST EQUIVALENT TO OTHER CREATION (IT IS)
    # DIR = '/home/aria/landslide/MDRUNS/IVAN_IGPS'
    # RESULTS = '/home/aria/landslide/RESULTS/GUIDELINES/NEW_ALGORITHM/test_list'
    # traj = jn(DIR, 'prot_apo_sim1_s10.dcd')
    # topo = jn(DIR, 'prot.prmtop')
    # apo1b_way1 = create_aanet(traj, topo=topo, selection='backbone', cutoff=5)
    # apo1b_way1.save(jn(RESULTS, 'apo1b_way1.p'))
    # apo1_way2 = create_aanet_multiselection(traj, ['backbone'], topo=topo, selection='all', cutoff=5,
    #                         output_atomic=jn(RESULTS, 'apo1_atomic.p'),
    #                         output_list=[jn(RESULTS, 'apo1b_way2.p')])
    # diff = nx.to_numpy_array(apo1b_way1.net) - nx.to_numpy_array(apo1_way2[0])
    # print(diff)
    # print(np.sum(np.abs(diff))) 
    # print(np.sum(np.abs(nx.to_numpy_array(apo1b_way1.net))))




######### TESTING IF CREATE_LIST WORKS
    DIR = '/home/aria/landslide/MDRUNS/IVAN_IGPS'
    apo_trajs = [jn(DIR, 'prot_apo_sim{0}_s10.dcd'.format(i)) for i in range(1,2)]
    sels = ['all', 'not hydrogen', 'backbone || name H HA', 'backbone', 'sidechain',
    'sidechain && not hydrogen', ['all', 'name H N']]
    outs = ['allH', 'all', 'backboneH', 'backbone', 'sidechainH', 
    'sidechain', 'amide_proton']
    RESULTS = '/home/aria/landslide/RESULTS/GUIDELINES/NEW_ALGORITHM'
    create_aanet_multiselection(apo_trajs,
                      selectionList=sels,                    
                      topo=jn(DIR, 'prot.prmtop'),
                      selection='all',
                      cutoff=5,
                      output_atomic=jn(RESULTS, 'atomic.p'),
                      output_list=[jn(RESULTS, out+'.p') for out in outs]
                    )




########OTHER OLD TESTS
# if __name__ == '__main__':
#     DIR = '/home/aria/landslide/MDRUNS/YEAST/all_trajs'
#     aanet1 = create_aanet(jn(DIR, 'model1_apo.dcd'), 
#                           topo='/home/aria/landslide/FRAMES/YEAST/ALLH/APO/frame_1.pdb',
#                           selection="not hydrogen",
#                           cutoff=5)
#     aanet1.save('apo1.p')
#     t10 = aanet1.apply_threshold(10)
#     nx.draw(t10)



#######OLD TEST TO PROVE EQUIVALENT TO OLDER ALGORITHM
#    apo1 = AANet(jn(DIR, 'model1_apo.dcd'), topo='/home/aria/landslide/FRAMES/YEAST/ALLH/APO/frame_1.pdb', selection="!(name =~ 'H.*')")
#    apo1_old = nx.read_gpickle('/home/aria/landslide/RESULTS/YEAST/1/aa_net/1.p')
#    print(nx.to_numpy_array(apo1.net), nx.to_numpy_array(apo1_old))
#    diff = nx.to_numpy_array(apo1.net)-nx.to_numpy_array(apo1_old)
#    diff_mask = diff[~np.eye(diff.shape[0],dtype=bool)].reshape(diff.shape[0],-1)
#    print(np.sum(np.abs(diff_mask)))
#    print(np.sum(np.abs(apo1.average)), np.sum(np.abs(nx.to_numpy_array(apo1_old))))