from dynpertnet import *
from maker import *
from os.path import join as jn

A2B1 = '/home/agheeraert/TRAJ_AMPK/A2B2/'

traj_list = [[jn(A2B1, 'APO', 'R'+str(i), 'a2b2_apo_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)],
             [jn(A2B1, 'HOLO', 'R'+str(i), 'a2b2+A769_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)],
             [jn(A2B1, 'HOLOATP', 'R'+str(i), 'a2b2+A769+ATP_R{0}.dry500ns.nc'.format(i)) for i in range(1,4)]]

topo_list = [jn(A2B1, 'APO', 'R1', 'A2B2_dry.prmtop'), 
	     jn(A2B1, 'HOLO', 'R1', 'A2B2+A769_dry.prmtop'),
	     jn(A2B1, 'HOLOATP', 'R1', 'A2B2+A769_ATP_dry.prmtop')]
	      
selections = ['protein', 'protein not hydrogen', 'backbone || name H HA', 'backbone', 'sidechain', 'sidechain && not hydrogen', ['protein', 'name H N']]

name_sels = ['allH', 'all', 'backboneH', 'backbone', 'sidechainH', 'sidechain', 'amideprot']

L_output = [jn('/home/agheeraert/TRAJ_AMPK/RESULTS/NEW_ALGO/A2B2', selection) for selection in name_sels]

name_list= ['apo', 'holo', 'holoatp']

L_output_atomic = [jn('/home/agheeraert/TRAJ_AMPK/RESULTS/NEW_ALGO/A2B2', '{0}.anpy'.format(name)) for name in name_list]

for selection, output_folder in zip(selections, L_output):
#	create_dpn_parallel(traj_list=traj_list, topo_list=topo_list, selection=selection, output_folder=output_folder, name_list=name_list)
    n_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=min(n_cpu, len(traj_list)))
    networks = pool.starmap(create_aanet_multiselection, zip(traj_list, [None]*len(traj_list), topo_list, ['all']*len(traj_list), [5]*len(traj_list), L_output_atomic))
