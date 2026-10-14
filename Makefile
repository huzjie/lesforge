PY ?= python

all: compile test doctor

compile:
	$(PY) -m compileall -q lesforge

test:
	$(PY) -m unittest discover -s tests -t .

doctor:
	$(PY) -m lesforge.doctor

bench:
	$(PY) -m lesforge.cli bench --config configs/mobile.yaml

serve:
	$(PY) -m lesforge.cli serve --config configs/mobile.yaml

.PHONY: all compile test doctor bench serve
