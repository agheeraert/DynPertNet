import numpy as np
from dynpertnet import t2o, three2one
from Bio.PDB import PDBParser
import pickle as pkl 

# L_OXY = [44, 16, 57.5, 50.6, 75.5, 75.7, 32.26, 61.28, 14.43]
# pdb_path = '/home/aria/landslide/FRAMES/YEAST/1/frame_1.pdb'

# pdb_path2 = '/home/aria/landslide/RESULTS/IGPS_TDEP/DATA/frame_1.pdb'
# L_OXY2 = [25.820999, 40.575001, 58.090000, 
#           67.162003, 40.580002, 69.556000,
#           34.550999, 55.306999, 32.634998]

pdb_path = '/home/aria/These/pdb/base_IGPS.pdb'

L_OXY = [-11.098, -12.387, 67.007004, 
        -7.896, 30.766001,	38.945,
        17.205,	-8.387,	69.603996]	


def get_pos_OXY(pdb_path, O, X, Y, output, chaintop=None):
    """ Draws the network using the O X Y method, O, X, Y are 3d coordinate points that should 
    frame the representation"""
    Ox = [c1-c2 for c1, c2 in zip(O, X)]   
    Oy = [c1-c2 for c1, c2 in zip(O, Y)]
    normOx = np.linalg.norm(Ox)  
    normOy = np.linalg.norm(Oy)
    structure = PDBParser().get_structure('X', pdb_path)[0]
    pos = {}
    distance_thresh = 1
    for atom in structure.get_atoms():
        if atom.id == 'CA':
            residue = atom.parent
            if chaintop:
                c = 1*(residue.parent.id == chaintop)
            else:
                c = 0
            AO = [c1-c2 for c1, c2 in zip(O, atom.coord)]
            x = np.linalg.norm(np.cross(AO,Ox))/normOx
            y = np.linalg.norm(np.cross(AO,Oy))/normOy+1*c
            pos[t2o(residue.resname)+str(residue.id[1])+':'+residue.parent.id] = (x, y)
    pkl.dump(pos, open(output, 'wb'))

if __name__ == '__main__':
#    get_pos_OXY(pdb_path, L_OXY[:3], L_OXY[3:6], L_OXY[6:], 'data/yeast_IGPS_pos.p')
#    get_pos_OXY(pdb_path2, L_OXY2[:3], L_OXY2[3:6], L_OXY2[6:], '/home/aria/landslide/RESULTS/IGPS_TDEP/DATA/pos2D.p')
    get_pos_OXY(pdb_path, L_OXY[:3], L_OXY[3:6], L_OXY[6:], '/home/aria/landslide/RESULTS/THEMA/pos2D.p')
