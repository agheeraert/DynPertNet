from pymol import cmd, stored
from pymol.cgo import *
import networkx as nx
from Bio.PDB.Polypeptide import aa1, aa3
three2one = dict(zip(aa3, aa1))
t2o = lambda X: three2one[X] if X in three2one else X[0] 
relabel = lambda X: t2o(X[:3])+X[3:-1]+':'+X[-1]
selection = lambda X: " or resi "+X[3:-1]+" and n. CA or n. C"



def drawNetwork(path, userSelection='all', r=1, edge_norm=None, alpha=0.5, 
                node_color=(0.6, 0.6, 0.6), edge_color1 = (1, 0, 0), 
                edge_color2 = (0, 0, 1), labelling='0',
                threshold=0):
    '''
    Draws a NetworkX network on the PyMol structure
    '''
    cmd.delete('nodes *edges')
    cmd.label(selection=userSelection, expression="")
    # Building position -- name correspondance
    stored.posCA = []
    stored.names = []
    userSelection = userSelection + " and n. CA or n. C"
    cmd.iterate_state(1, selector.process(userSelection), "stored.posCA.append([x,y,z])")
    cmd.iterate(userSelection, 'stored.names.append(resn+resi+chain)')
    stored.labels = list(map(relabel, stored.names))
    stored.resid = list(map(selection, stored.names))
    node2id = dict(zip(stored.labels, stored.resid))
    node2CA = dict(zip(stored.labels, stored.posCA))
    # Getting graph
    net = nx.read_gpickle(path)

    cmd.set('auto_zoom', 0)
    cmd.set("cgo_sphere_quality", 4)
    #Drawing nodes
    # selnodes = ""
    # for u in net.nodes():
    #     x, y, z = node2CA[u]
    #     obj+=[SPHERE, x, y, z, r]
    #     selnodes+=node2id[u]
    # selnodes = selnodes[4:]

    # if labelling=='1':
    #     cmd.label(selection=selnodes, expression="t2o(resn)+resi")
    # if labelling=='3':
    #     cmd.label(selection=selnodes, expression="resn+resi")
    # cmd.load_cgo(obj, 'nodes')

    if edge_norm == None:
        edge_norm = max([net.edges()[(u, v)]['weight'] for u, v in net.edges()])/r
    obj1, obj2, nodelist = [], [], []
    for u, v in net.edges():
        radius = net[u][v]['weight']/edge_norm
        if abs(net[u][v]['weight']) >= threshold:
            if 'color' in net[u][v]: 
                if net[u][v]['color'] == 'r':
                    obj1+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color1, *edge_color1]
                else:
                    obj2+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color2, *edge_color2]
            else:
                if net[u][v]['weight'] <= 0:
                    obj1+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color1, *edge_color1]
                else:
                    obj2+=[CYLINDER, *node2CA[u], *node2CA[v], radius, *edge_color2, *edge_color2]

            nodelist+=[u, v]

    #Drawing nodes
    obj=[COLOR, *node_color]
    nodelist = set(nodelist)
    selnodes = ''.join([node2id[u] for u in nodelist])[4:]
    for u in nodelist:
        x, y, z = node2CA[u]
        obj+=[SPHERE, x, y, z, r]

    if labelling=='1':
        cmd.label(selection=selnodes, expression="t2o(resn)+resi")
    if labelling=='3':
        cmd.label(selection=selnodes, expression="resn+resi")
    cmd.load_cgo(obj, 'nodes')


    cmd.load_cgo(obj1, 'holo_edges')
    cmd.load_cgo(obj2, 'apo_edges')

        # obj = [
        # 'ALPHA', alpha,
        # 'COLOR', node_color[0], node_color[1], node_color[2],
        # 'SPHERE', float(x), float(y), float(z), float(r),
        # ]
        # cmd.load_cgo(obj, 'sphere')

cmd.extend("drawNetwork", drawNetwork)
cmd.extend("delNet", lambda: cmd.delete('nodes *edges'))
cmd.extend("t2o", lambda X: three2one[X] if X in three2one else X[0])
