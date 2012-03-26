class InputError(Exception):
	def __init__(self,msg):
		self.msg = msg
	def __str__(self):
		return repr(self.value)
class ModuleNotLoadedError(Exception):
	def __init__(self,msg):
		self.msg = msg
	def __str__(self):
		return repr(self.value)