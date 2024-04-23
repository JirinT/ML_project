# Machine Learning Project

Dataset link: https://www.repository.cam.ac.uk/items/6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8

### Loss functions:
1: CrossEntropyLoss <br>
2: MSE Loss <br>
3: L1 Loss <br>
4: NLL Loss <br>

### Optimizer:
1: Adam <br>
2: Stochastic Gradient Descent <br>

## Get Started
1. Clone git repo: <br>
   ```
   git clone https://github.com/JirinT/ML_project.git
   ```
2. Create virtual env from requirements file: <br>
   ```
   python3 -m venv .venv
   ```
   ```
   .venv\Scripts\activate
   ```
   ```
   python3 -m pip install -r requirements.txt
   ```
3. Open <em>file_downloader.py</em>
4. Change <em>download_dir</em> to own preference and execute <em>file_downloader.py</em>
5. Select configuration for training in <em>config.py</em>
6. Execute <em>train.py</em> to train the model
7. Select <em>model_path_to_load</em> in <em>config.py</em> and execute <em>test.py</em> to test the model
