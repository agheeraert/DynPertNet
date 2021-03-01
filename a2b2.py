from dynpertnet import *
from maker import *
from os.path import join as jn

INPUT_FOLDER = '/home/agheeraert/TRAJ_AMPK/A2B2/'
OUTPUT_FOLDER = '/home/agheeraert/TRAJ_AMPK/RESULTS/NEW_ALGO/A2B2'

traj_list = [[jn(INPUT_FOLDER, 'APO', 'R'+str(i), 'a2b2_apo_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)],
             [jn(INPUT_FOLDER, 'HOLO', 'R'+str(i), 'a2b2+A769_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)],
             [jn(INPUT_FOLDER, 'HOLOATP', 'R'+str(i), 'a2b2+A769+ATP_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)]]

topo_list = [jn(INPUT_FOLDER, 'APO', 'R1', 'A2B2_dry.prmtop'), 
	     jn(INPUT_FOLDER, 'HOLO', 'R1', 'A2B2+A769_dry.prmtop'),
	     jn(INPUT_FOLDER, 'HOLOATP', 'R1', 'A2B2+A769_ATP_dry.prmtop')]
	      
selections = ['protein', 'protein not hydrogen', 'backbone || name H HA', 'backbone', 'sidechain', 'sidechain && not hydrogen', ['protein', 'name H N']]

name_sels = ['allH', 'all', 'backboneH', 'backbone', 'sidechainH', 'sidechain', 'amideprot']

L_output = [jn(OUTPUT_FOLDER, selection) for selection in name_sels]

name_list= ['apo', 'holo', 'holoatp']

for replica in range(1,4):
    for selection, output_folder in zip(selections, L_output):
        n_cpu = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=min(n_cpu, len(traj_list)))
        new_trajlist = [traj[replica] for traj in traj_list]
        print(new_trajlist)
        L_output_atomic = [jn(OUTPUT_FOLDER, '{0}_{1}.anpy'.format(name, replica)) for name in name_list]
        networks = pool.starmap(create_aanet_multiselection, zip(new_trajlist, [None]*len(traj_list), topo_list, ['all']*len(traj_list), [5]*len(traj_list), L_output_atomic))
