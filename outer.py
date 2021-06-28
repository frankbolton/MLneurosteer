import inner

for r in range(10):
	print(f"inside the loop - value == {r}")
	inner.run(r)
