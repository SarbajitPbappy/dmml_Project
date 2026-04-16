.PHONY: venv install train eval explain api dashboard

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && python -m pip install -r requirements.txt

train:
	. .venv/bin/activate && python -m retrox.cli.train --city sj --horizon 4

eval:
	. .venv/bin/activate && python -m retrox.cli.evaluate --city sj --horizon 4

explain:
	. .venv/bin/activate && python -m retrox.cli.explain --city sj --horizon 4

api:
	. .venv/bin/activate && uvicorn retrox.api.main:app --reload

dashboard:
	. .venv/bin/activate && streamlit run retrox/dashboard/app.py

