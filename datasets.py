import json
from torch.utils.data import Dataset
from pathlib import Path
import h5py
from dependency_tree import *
import numpy as np
from tqdm import tqdm
import torch
data_type = "train"

class CONLLUDataSet:
	'''
	[
	{
		"text": Sentence constructed from col1 of conllu.
		"tokens":
			[
				{
					"id": token id,
					"form": token original form,
					"head":id of the token's head,
					"head_form": form of the token's head
				}
			]
	}]
	'''
	def __init__(self, path=None):
		self.data = None
		if path is not None:
			path = Path(path)
			if path.suffix == ".json":
				self.load_processed_data(path)
			elif path.suffix == ".conllu":
				self.load_raw_data(path)

	def load_raw_data(self, path):
		text_buffer = []
		data = []
		data_item_buffer = {"text": "", "tokens": []}
		
		with open(path, "r") as f:
			for line in f:
				if line.startswith("#"):
					continue
				if line.strip():
					id, form, head = [line.split("\t")[i] for i in [0, 1, 6]]
					try:
						int(id)
					except ValueError:
						continue
					data_item_buffer["tokens"].append({
						"id": id, 
						"form": form, 
						"head": head
					})
					text_buffer.append(form)
					#到这里就读完了一个句子的dependency parsing结果
				else:
					# 补充 head_form
					if len(data_item_buffer['tokens']) >= 2:
						id_to_form = {t["id"]: t["form"] for t in data_item_buffer["tokens"]}
						for token in data_item_buffer["tokens"]:
							token["head_form"] = "ROOT" if token["head"] == "0" else id_to_form.get(token["head"], "_")
						
						data_item_buffer["text"] = " ".join(text_buffer)
						text_buffer = []
						data.append(data_item_buffer)
						data_item_buffer = {"text": "", "tokens": []}
			
			# 如果到了末尾非空行，text_buffer还有内容，就做相应处理
			if text_buffer:
				if len(data_item_buffer['tokens']) >= 2:
					id_to_form = {t["id"]: t["form"] for t in data_item_buffer["tokens"]}
					for token in data_item_buffer["tokens"]:
						token["head_form"] = "ROOT" if token["head"] == "0" else id_to_form.get(token["head"], "_")
					
					data_item_buffer["text"] = " ".join(text_buffer)
					data.append(data_item_buffer)
		
		self.data = data

	def load_processed_data(self, path):
		self.data = json.load(open(path, 'r'))

	def save_data_to_json(self):
		json.dump(self.data, open("./data/data.json", "w", encoding="UTF-8"))
		
class DependencyH5DataWriter:
	@staticmethod
	def get_mean_states(hidden_states_dataset):
		'''
		hidden_states_dataset is an instance of h5py._hl.dataset.Dataset
		'''
		return np.mean(np.stack([hs for hs in hidden_states_dataset], axis=0), axis=0)

	def __init__(self, depedency_data_path=None, hidden_states_path=None):
		self.dependency_trees=None
		if hidden_states_path is not None and depedency_data_path is None:
			raise ValueError("dependency_data_path must be set when hidden_states_path is set")

	def load_dependency_trees(self, dependency_data_path):
		dependency_data = CONLLUDataSet(dependency_data_path)
		self.dependency_trees = [Tree(tree_dict) for tree_dict in dependency_data.data]
	
	def load_hidden_states_and_write(self, hidden_states_path):
		assert self.dependency_trees is not None
		with h5py.File(hidden_states_path) as r:
			with h5py.File(f"dependency_{data_type}_data.h5", 'w') as w:
				shape = r["/sentence_000000"]['hidden_states'].shape[1:]
				depth_group = w.create_group("word_depth")
				distance_group = w.create_group("word_distance")
				depth_group.create_dataset(
					"X",
					shape=(0,) + shape,
					maxshape=(None,) + shape,
					dtype='float32',
					chunks=True,
					compression="gzip"
				)
				depth_group.create_dataset(
					"depth",
					shape=(0,),
					maxshape=(None, ),
					dtype='int16',
					chunks=True
				)

				distance_group.create_dataset(
					"X_diff",
					shape=(0,) + shape,
					maxshape=(None,) + shape,
					dtype='float32',
					chunks=True,
					compression="gzip"
				)
				distance_group.create_dataset(
					"distance",
					shape=(0,),
					maxshape=(None, ),
					dtype='int16',
					chunks=True
				)
				for sent_group in tqdm(r.values(), desc=f"generating {data_type} dataset:"):
					tree = self.dependency_trees[int(sent_group.name[-6:])]
					if len(tree.nodes) < 3:
						print(f"单词数量太少了:{[node.value for node in tree.nodes.values()]}")
						continue
					ids_in_tree = np.vectorize(lambda s: s.decode("UTF-8"))(sent_group['id_in_tree'][:])
					wi, wj = np.meshgrid(ids_in_tree, ids_in_tree, indexing='ij')
					W_ij = np.stack([wi, wj], axis=-1)
					mask = W_ij[..., 0] <= W_ij[..., 1] #i<=j取得到等号是方便后面损失的池化
					hidden_states = sent_group['hidden_states'][:]
					diff_states = hidden_states[:, None, :, :] - hidden_states[None, :, :, :]
					#得到i <= j的词对和差分hidden states
					W_ij = W_ij[mask]
					diff_states = diff_states[mask]
					distance_batch = []
					X_diff_batch = []
					depth_batch = [tree.nodes[id].depth() for id in ids_in_tree]
					# X_batch = hidden_states
					depth_group['depth'].resize(depth_group['depth'].shape[0] + len(depth_batch), axis=0)
					depth_group['depth'][-len(depth_batch):] = depth_batch

					depth_group['X'].resize(depth_group['X'].shape[0] + hidden_states.shape[0], axis=0)
					depth_group['X'][-hidden_states.shape[0]:] = hidden_states


					for pair, diff in zip(W_ij, diff_states):
						node_a = tree.nodes[pair[0]]
						node_b = tree.nodes[pair[1]]
						distance_batch.append(tree.compute_distance(node_a, node_b))
						X_diff_batch.append(diff)
					distance_group['distance'].resize(distance_group['distance'].shape[0] + len(distance_batch), axis=0)
					distance_group['distance'][-len(distance_batch):] = distance_batch

					distance_group['X_diff'].resize(distance_group['X_diff'].shape[0] + len(X_diff_batch), axis=0)
					distance_group['X_diff'][-len(X_diff_batch):] = X_diff_batch



class DepthDataSet(Dataset):
	def __init__(self, h5_path):
		self.h5_path = h5_path
		with h5py.File(h5_path, "r") as f:
			self.length = len(f['word_depth']['X'])
		self.file = None

	def _open(self, path):
		if self.file is None:
			self.file = h5py.File(self.h5_path, "r")

	def __len__(self):
		return self.length
	
	def __getitem__(self, idx, layer_index=5):
		self._open(self.h5_path)
		#select method to select the specific transformer layer outputs
		X = torch.from_numpy(self.file['word_depth']['X'][idx]).select(0, layer_index)
		Y = torch.tensor(self.file['word_depth']['depth'][idx], dtype=torch.short)
		return X, Y
	
class DistanceDataSet(Dataset):
	def __init__(self, h5_path):
		self.h5_path = h5_path
		with h5py.File(h5_path, "r") as f:
			self.length = len(f['word_distance']['X_diff'])
		self.file = None

	def _open(self, path):
		if self.file is None:
			self.file = h5py.File(self.h5_path, "r")

	def __len__(self):
		return self.length
	
	def __getitem__(self, idx, layer_index=5):
		self._open(self.h5_path)
		X = torch.from_numpy(self.file['word_distance']['X_diff'][idx]).select(0, layer_index)
		Y = torch.tensor(self.file['word_distance']['distance'][idx], dtype=torch.short)
		return X, Y

