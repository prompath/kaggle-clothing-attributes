notebook:
	jupytext --to ipynb --output $(target) $(source)
	jupyter nbconvert --to notebook --inplace --execute $(target)

help:
	@echo "notebook        - Convert a python script to a jupyter notebook"
	@echo "                and execute all cells"
