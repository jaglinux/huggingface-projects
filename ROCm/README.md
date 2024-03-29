https://rocm.blogs.amd.com/artificial-intelligence/llama2-lora/README.html <br>

pip install -q pandas torch peft transformers==4.31.0 trl accelerate scipy <br>
git clone --recurse https://github.com/ROCmSoftwarePlatform/bitsandbytes <br>
cd bitsandbytes/ <br>
git checkout 4c0ca08aa24d622940d9abdcff6090efc85dbc30 <br>
make hip <br>
python setup.py install <br>
pip install tensorboardX <br>
