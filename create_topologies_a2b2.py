import mdtraj as md
from os.path import join as jn
import pickle as pkl

A2B1 = '/home/agheeraert/TRAJ_AMPK/A2B2/'

traj_list = [[jn(A2B1, 'APO', 'R'+str(i), 'a2b2_apo_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)],
[jn(A2B1, 'HOLO', 'R'+str(i), 'a2b2+A769_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)],
[jn(A2B1, 'HOLOATP', 'R'+str(i), 'a2b2+A769+ATP_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)]]

topo_list = [jn(A2B1, 'APO', 'R1', 'A2B2_dry.prmtop'), 
jn(A2B1, 'HOLO', 'R1', 'A2B2+A769_dry.prmtop'),
jn(A2B1, 'HOLOATP', 'R1', 'A2B2+A769_ATP_dry.prmtop')]

name_list= ['apo', 'holo', 'holoatp']

L_output_atomic = [jn('/home/agheeraert/TRAJ_AMPK/RESULTS/NEW_ALGO/A2B2', '{0}.topy'.format(name)) for name in name_list]

for i in range(len(name_list)):
    t = md.load_frame(traj_list[i][0], 0, top=topo_list[i])
    topo = t.topology
    pkl.dump(topo, open(L_output_atomic[i], 'wb'))

    if str(next(topo.residues))[-1] == '0':
        for res in t.topology.residues:
            print(res.resSeq)
            res.resSeq += 1
            print(res.resSeq)
    t.save(L_output_atomic[i].replace('.topy', '.pdb'))
