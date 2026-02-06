# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
clean: Makefile
	@rm -rf source/examples/*.zip
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: 
	@cd source/examples && find * -maxdepth 0 -type d | xargs -I {} sh -c ' \
		echo "zipping {}..."; \
		zip -q -r {}.zip {} \
			-x "*.pyc" \
			-x "*__pycache__*" \
			-x "*pre_process.log" \
			-x "quick_start_example/output/*" \
			-x "quick_start_example/pulsar2-run-helper/models/*.axmodel" \
			-x "quick_start_example/pulsar2-run-helper/sim_inputs/0/*.bin" \
			-x "quick_start_example/pulsar2-run-helper/sim_outputs/0/*.bin" \
	'
	@echo "sphinx build..."
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
