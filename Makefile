notebook:
	jupytext --to ipynb --output $(target) $(source)
	jupyter nbconvert --to notebook --inplace --execute $(target)
