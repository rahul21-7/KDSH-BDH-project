import torch
import pandas as pd
import os

from bdh import BDH, BDHConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on {device.upper()} Mode ---")

#1. setup model
config = BDHConfig()
model = BDH(config).to(device)

def get_book_memory(file_path, cache_dir="memory_cache"):
    # 1. Check if we already processed this book
    os.makedirs(cache_dir, exist_ok=True)
    book_name = os.path.basename(file_path).replace(".txt", "")
    cache_path = os.path.join(cache_dir, f"{book_name}.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached memory for: {book_name}")
        return torch.load(cache_path, map_location=device)
    
    # 2. If not cached, process the novel   
    state = None
    bytes_read = 0
    file_size = os.path.getsize(file_path)
    print(f"Processing new novel: {book_name}...")
    with torch.no_grad():
        with open(file_path, "rb") as f:    #read as bytes
            while True:
                chunk = f.read(1024)        #1024 bytes at a time
                if not chunk: break

                #bytes->tensor
                idx = torch.tensor([list(chunk)]).to(model.lm_head.device)

                #pass the memory state forward
                _, _, state, tension =  model(idx, prev_state = state)

                bytes_read += len(chunk)
                print(f"  Progress: {(bytes_read/file_size)*100:.1f}%", end="\r")
    
    # 3. Save it for next time
    print(f"\nSaving memory to {cache_path}...")
    torch.save(state, cache_path)

    return state

def evaluate_dataset(csv_path):
    df = pd.read_csv(csv_path)
    results = []

    book_memories = {}

    for _, row in df.iterrows():
        book_name = row["book_name"]
        backstory = row["content"]

        #1. load book into memory if it hasn't been read already
        if book_name not in book_memories:
            print(f"Reading full novel: {book_name}...")
            book_memories[book_name] = get_book_memory(f"../Dataset/Books/{book_name}.txt")

        novel_state = book_memories[book_name]

        # 2. Test the backstory against that memory
        backstory_idx = torch.tensor([list(backstory.encode('utf-8'))]).to(device)

        with torch.no_grad():
            # We use the backstory as the target to get the Loss
            _, loss, _, tension = model(backstory_idx, targets=backstory_idx, prev_state=novel_state)
        
        results.append({
            "id": row['id'],
            "loss": loss.item(),
            "tension": tension.item()
        })

    return pd.DataFrame(results)

# Run it
train_analysis_df = evaluate_dataset("../Dataset/train.csv")
train_analysis_df.to_csv("train_analysis.csv", index=False)

test_analysis_df = evaluate_dataset("../Dataset/test.csv")
train_analysis_df.to_csv("test_analysis.csv", index=False)

