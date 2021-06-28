import neptune.new as neptune



print("inner is running")
def run(index):
	run = neptune.init(project='frankbolton/helloworld') # your crdentials
	print(f"inside the 'inner' loop- index == {index}")

	params = {
		"index": index
	}

	run["parameters"] = params
	run['JIRA'] = "NPT-954"
	run.stop()
