# Supabase preparation:
1. build a project
2. enable ```vector``` type file by click ```Database -> Extensions -> search "vector"``` and enable the extensions
3. make a table ```documents```, cheklist the ```Enable Row Level Security``` and set with a column :
```
id : int8; -> primary key
content : text;
embedding : vector;
metadate : jsonb;
type : text;
chunk_id : uuid
```
5. disable the RLS (Row Level Security) by clicking ```Add RLS policy -> 
Disable RLS -> Confirm```
6. make a table ```tables```, cheklist the ```Enable Row Level Security``` and set with a column :
```
chunk_id : uuid -> primary key
table_data : jsonb;
embedding : vector;
metadata : jsonb;
description : text
```
7. disable the RLS (Row Level Security) by clicking ```Add RLS policy -> 
Disable RLS -> Confirm```

# Run the code :
1. install the ```requirements.txt``` first in your environment
2. fill the ```.env``` supabase url and the API key
3. make sure the input and output folder in the same folder as code and other dependencies
4. If the venv supports GPU, make sure install the PyTorch with the correct CUDA version. (my system supports CUDA 12.5, so i run this ```**pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121**```)
5. if find some error while loading the model, make sure to test your env supported GPU or not, by running the ```test.py``` first. Expected result :
   ```
   "CUDA Available: True"
   "GPU Name: NVIDIA GeForce RTX 3050".
   ```
   if you have GPU but its not detected, run ```nvidia-smi``` in the cmd first
7. the code also can be run in the CPU
