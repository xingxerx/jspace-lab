import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("mem_GB", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as e:
    print("transformers MISSING:", e)
