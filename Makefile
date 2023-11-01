.PHONY: hpc_env local_env data

hpc_env:
	if [ ! -d "env" ]; then \
		@module load python3/3.10.11; \
		python3 -m venv env; \
	fi
	@source env/bin/activate; \
	python3 -m pip install -r requirements.txt

local_env:
	if [ ! -d "env" ]; then \
		python3 -m venv env; \
	fi
	@source env/bin/activate; \
	python3 -m pip install -r requirements.txt

