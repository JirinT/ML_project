Dataset link: https://www.repository.cam.ac.uk/items/6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8

Get Started:
	1. Create virtual env from requirements file:

		python3 -m venv .venv
		.venv\Scripts\activate
		python -m pip install -r requirements.txt

	2. Open file_downloader.py
	3. Change download_dir to own preference and execute file_downloader.py - it will download the dataset (Aprox. 65GB)
	4. Select configuration for training in config.py
	5. Execute train.py to train and save the model
	6. Select model_path_to_load in config.py and execute test.py to test the model