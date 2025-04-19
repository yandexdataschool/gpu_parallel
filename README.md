# gpu_parallel
Roman Garipov's gpu_parallel.py from hogwild_inference

**Usage:**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ^-- gpu_parallel will split these between workers
python3 gpu_parallel.py --start 0 --end 128 --use_queue --script example_script.py --extra_args "--save_folder $SNAPSHOT_PATH"
```


For Yandex staff: here's an example usage in our infrastructure: https://nda.ya.ru/t/71dLrS2y7DmWrz
