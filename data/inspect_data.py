from datasets import load_dataset

ds = load_dataset("Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset", split="train")
print(ds)
print("columns:", ds.column_names)

ex = ds[0]
for k, v in ex.items():
    print(f"\n===== {k} ({type(v).__name__}) =====")
    print(repr(v)[:1500])