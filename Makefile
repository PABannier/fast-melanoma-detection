run-app:
	@streamlit run deployment/app.py

run-container:
	@docker build .

train-model:
