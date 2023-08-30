import ROOT

import numpy as np
import graph_nets as gn
from graph_nets import utils_tf
import sonnet as snt
import time

# defining graph properties
num_nodes=5
num_edges=20
snd = np.array([1,2,3,4,2,3,4,3,4,4,0,0,0,0,1,1,1,2,2,3])
rec = np.array([0,0,0,0,1,1,1,2,2,3,1,2,3,4,2,3,4,3,4,4])
node_size=4
edge_size=4
global_size=1
LATENT_SIZE = 100
NUM_LAYERS = 4
processing_steps = 5

# method for returning dictionary of graph data
def get_graph_data_dict(num_nodes, num_edges, NODE_FEATURE_SIZE=2, EDGE_FEATURE_SIZE=2, GLOBAL_FEATURE_SIZE=1):
    return {
      "globals": 10*np.random.rand(GLOBAL_FEATURE_SIZE).astype(np.float32)-5.,
      "nodes": 10*np.random.rand(num_nodes, NODE_FEATURE_SIZE).astype(np.float32)-5.,
      "edges": 10*np.random.rand(num_edges, EDGE_FEATURE_SIZE).astype(np.float32)-5.,
      "senders": snd, 
      "receivers": rec
    }

# method to instantiate mlp model to be added in GNN
def make_mlp_model():
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True),
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

# defining GraphIndependent class with MLP edge, node, and global models.
class MLPGraphIndependent(snt.Module):
  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = gn.modules.GraphIndependent(
        edge_model_fn = lambda: snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True),
        node_model_fn = lambda: snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True),
        global_model_fn = lambda: snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True))

  def __call__(self, inputs):
    return self._network(inputs)

# defining Graph network class with MLP edge, node, and global models.
class MLPGraphNetwork(snt.Module):
  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    self._network = gn.modules.GraphNetwork(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model)

  def __call__(self, inputs):
    return self._network(inputs)

# defining a Encode-Process-Decode module for LHCb toy model
class EncodeProcessDecode(snt.Module):

  def __init__(self,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent()
    self._core = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
    self._output_transform = MLPGraphIndependent()

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops


# Instantiating EncodeProcessDecode Model
ep_model = EncodeProcessDecode()

# Initializing randomized input data
GraphData = get_graph_data_dict(num_nodes,num_edges, node_size, edge_size, global_size)

#input_graphs  is a tuple representing the initial data
input_graph_data = utils_tf.data_dicts_to_graphs_tuple([GraphData])

# Initializing randomized input data for core
# note that the core network has as input a double number of features
CoreGraphData = get_graph_data_dict(num_nodes, num_edges, 2*LATENT_SIZE, 2*LATENT_SIZE, 2*LATENT_SIZE)
input_core_graph_data = utils_tf.data_dicts_to_graphs_tuple([CoreGraphData])

#initialize graph data for decoder (input is LATENT_SIZE)
DecodeGraphData = get_graph_data_dict(num_nodes,num_edges, LATENT_SIZE, LATENT_SIZE, LATENT_SIZE)

# Make prediction of GNN
output_gn = ep_model(input_graph_data, processing_steps)
print("---> Input:\n",input_graph_data)
print("\n\n------> Input core data:\n",input_core_graph_data)
print("\n\n---> Output:\n",output_gn)

# Make SOFIE Model
encoder = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._encoder._network, GraphData, filename = "encoder")
encoder.Generate()
encoder.OutputGenerated()

core = ROOT.TMVA.Experimental.SOFIE.RModel_GNN.ParseFromMemory(ep_model._core._network, CoreGraphData, filename = "core")
core.Generate()
core.OutputGenerated()

decoder = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._decoder._network, DecodeGraphData, filename = "decoder")
decoder.Generate()
decoder.OutputGenerated()

output_transform = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._output_transform._network, DecodeGraphData, filename = "output_transform")
output_transform.Generate()
output_transform.OutputGenerated()

# Compile now the generated C++ code from SOFIE
ROOT.gInterpreter.Declare('#pragma cling optimize(2)')
ROOT.gInterpreter.Declare('#include "encoder.hxx"')
ROOT.gInterpreter.Declare('#include "core.hxx"')
ROOT.gInterpreter.Declare('#include "decoder.hxx"')
ROOT.gInterpreter.Declare('#include "output_transform.hxx"')

#helper function to print SOFIE GNN data structure
def PrintSofie(output, printShape = False):
    n = np.asarray(output.node_data)
    e = np.asarray(output.edge_data)
    g = np.asarray(output.global_data)
    if (printShape) :
        print("SOFIE data ... shapes",n.shape,e.shape,g.shape)
    print(" node data", n.reshape(n.size,))
    print(" edge data", e.reshape(e.size,))
    print(" global data",g.reshape(g.size,))

def CopyData(input_data) :
  output_data = ROOT.TMVA.Experimental.SOFIE.Copy(input_data)
  return output_data

# Build  SOFIE GNN Model and run inference
class  SofieGNN:
    def __init__(self):
        self.encoder_session = ROOT.TMVA_SOFIE_encoder.Session()
        self.core_session = ROOT.TMVA_SOFIE_core.Session()
        self.decoder_session = ROOT.TMVA_SOFIE_decoder.Session()
        self.output_transform_session = ROOT.TMVA_SOFIE_output_transform.Session()

    def infer(self, graphData):
        # copy the input data
        input_data = CopyData(graphData)

        # running inference on sofie
        self.encoder_session.infer(input_data)
        latent0 = CopyData(input_data)
        latent = input_data
        output_ops = []
        for _ in range(processing_steps):
            core_input = ROOT.TMVA.Experimental.SOFIE.Concatenate(latent0, latent, axis=1)
            self.core_session.infer(core_input)
            latent = CopyData(core_input)
            self.decoder_session.infer(core_input)
            self.output_transform_session.infer(core_input)
            output = CopyData(core_input)
            output_ops.append(output)
            
        return output_ops
    
# Test both GNN on some simulated events
def GenerateData(): 
    data = get_graph_data_dict(num_nodes,num_edges, node_size, edge_size, global_size)
    return data

numevts = 100
dataSet = []
for i in range(0,numevts): 
    data = GenerateData()
    dataSet.append(data)

# Run graph_nets model
# First we convert input data to the required input format 
gnetData = []
for i in range(0,numevts):
    graphData = dataSet[i]
    gnet_data_i = utils_tf.data_dicts_to_graphs_tuple([graphData])
    gnetData.append(gnet_data_i)

# Function to run the graph net
def RunGNet(inputGraphData) :
    output_gn = ep_model(inputGraphData, processing_steps)
    return output_gn

start = time.time()
hG = ROOT.TH1D("hG","hG",100,1,0)
for i in range(0,numevts): 
    out = RunGNet(gnetData[i])
    g = out[1].globals.numpy()
    hG.Fill(np.mean(g))
    
end = time.time()
print("elapsed time for ",numevts,"events = ",end-start)
hG.Draw()
ROOT.gPad.Draw()

# running SOFIE-GNN
sofieData = []
for i in range(0,numevts):
    graphData = dataSet[i]
    input_data = ROOT.TMVA.Experimental.SOFIE.GNN_Data()
    input_data.node_data = ROOT.TMVA.Experimental.AsRTensor(graphData['nodes'])
    input_data.edge_data = ROOT.TMVA.Experimental.AsRTensor(graphData['edges'])
    input_data.global_data = ROOT.TMVA.Experimental.AsRTensor(graphData['globals'])
    sofieData.append(input_data)

hS = ROOT.TH1D("hS","hS",100,1,0)
gnn = SofieGNN()
start = time.time()
for i in range(0,numevts): 
    #print("inference event....",i)
    out = gnn.infer(sofieData[i])
    g = np.asarray(out[1].global_data)
    hS.Fill(np.mean(g))
    
end = time.time()
print("elapsed time for ",numevts,"events = ",end-start)
hS.Draw()
ROOT.gPad.Draw()
