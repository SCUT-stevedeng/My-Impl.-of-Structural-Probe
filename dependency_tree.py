class TreeNode:
	def __init__(self, id:str, value:str):
		self.id=id
		self.value=value
		self.parent=None
		self.children=[]

	def add_child(self, child:"TreeNode"):
		child.parent=self
		self.children.append(child)

	def depth(self):
		#这里练习使用递归
		if self.parent:
			return 1 + self.parent.depth() 

		else:
			return 0
		
	def get_node_from_form(self, form):
		if self.value==form:
			return self
		else:
			for child in self.children:
				found = child.get_node_from_form(form)
				if found:
					return found
			return None
		

class Tree:
	def __init__(self, data_dict=None):
		self.nodes = None
		self.root = None
		if data_dict:
			self.build_tree(data_dict)

	def build_tree(self, data_dict):
		self.nodes={}
		for token in data_dict["tokens"]:
			if token["head"] not in self.nodes:
				self.nodes[token["head"]] = TreeNode(token['head'], token['head_form'])
			if token['id'] not in self.nodes:
				self.nodes[token['id']] = TreeNode(token['id'], token['form'])
		
		for token in data_dict['tokens']:
			self.nodes[token['id']].parent = self.nodes[token['head']]
			self.nodes[token['head']].children.append(self.nodes[token['id']])

		self.root = self.nodes['0']

	def print_tree(self):

		def dfs(node, prefix="", is_last=True):
			connector = "└── " if is_last else "├── "
			print(prefix + connector + f"{node.value} ({node.id})")

			new_prefix = prefix + ("    " if is_last else "│   ")

			for i, child in enumerate(node.children):
				dfs(
					child,
					new_prefix,
					i == len(node.children) - 1
				)

		if self.root:
			dfs(self.root)

	def find_lca(self, node_a, node_b):
		cur_node = node_a
		path_a = {cur_node}
		#这里练习使用循环
		while cur_node.parent:
			path_a.add(cur_node.parent)
			cur_node = cur_node.parent
		
		index_node = node_b
		while index_node not in path_a:
			index_node = index_node.parent
		return index_node
	
	def compute_distance(self, node_a, node_b):
		return node_a.depth() + node_b.depth() - 2*self.find_lca(node_a, node_b).depth()
	