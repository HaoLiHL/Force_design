import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.multiprocessing import Pool, cpu_count

def worker_function(arg1, arg2):
    # Do some computation here
    result = arg1 + arg2
    return result

if __name__ == '__main__':
    # Generate input data
    data1 = torch.randn(100)
    data2 = torch.randn(100)
    input_data = [(data1[i], data2[i]) for i in range(100)]

    # Create DataLoader and TensorDataset
    dataset = TensorDataset(data1, data2)
    dataloader = DataLoader(dataset, batch_size=50, num_workers=cpu_count())

    # Create a Pool of workers
    pool = Pool(processes=cpu_count())

    # Map the worker function to the input data
    output_data = pool.starmap(worker_function, input_data)

    # Print the first 10 results
    print(output_data[:10])
