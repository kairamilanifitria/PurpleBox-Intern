# 2_Image Description

To run this:
1. install the requirements.txt first in your environment
2. make sure the input and output folder in the same folder as code and other depedencies
3. If the env supports GPU, make sure install the **PyTorch with the correct CUDA version**. (my sistem supports system supports CUDA 12.5, so i run this "**pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121**")
4. if find some error while loading the model, make sure to test your env supported GPU or not, by running the "test.py" first.
   Expected result :
  "CUDA Available: True"
  "GPU Name: NVIDIA GeForce RTX 3050"
   if you have GPU but its not detected, run "**nvidia-smi**" in the **cmd** first
5. the code also can be run in the CPU, but it may takes longer time to process
6. some arguments errors were solved this time, but maybe it can show problems in another env, so make sure all of them are run in the same env
