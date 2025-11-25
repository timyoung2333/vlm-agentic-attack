import json
import random
import os
from collections import defaultdict

def get_subset(input_path, output_path):
    """
    Reads the combined.json file, groups items by their 'Type' (intent category),
    and randomly samples 10 items from each category.
    """
    print(f"Reading from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    subset_data = {}
    category_keywords = {'chat': 'Type', 'embodied': 'category'}
    # Process both 'chat' and 'embodied' sections if they exist
    for section in ['chat', 'embodied']:
        if section not in data:
            continue
            
        items = data[section]
        print(f"Processing section: {section}, Total items: {len(items)}")
        
        # Group by Type
        grouped = defaultdict(list)
        for item in items:
            # Use 'Type' as the category. If missing, use 'unknown'

            category_key = category_keywords.get(section, 'Type')
            category = item.get(category_key, 'unknown')
            grouped[category].append(item)
        
        sampled_items = []
        for category, group_items in grouped.items():
            count = len(group_items)
            if count > 10:
                print(f"  Category '{category}': Sampling 10 from {count} items")
                sampled_items.extend(random.sample(group_items, 10))
            else:
                print(f"  Category '{category}': Taking all {count} items (<= 10)")
                sampled_items.extend(group_items)
        
        subset_data[section] = sampled_items
        
    # Save the subset
    with open(output_path, 'w') as f:
        json.dump(subset_data, f, indent=4)
    
    print(f"Subset saved to {output_path}")

if __name__ == "__main__":
    # Define paths assuming script is run from project root or mvp folder
    # Try to locate the data file
    possible_paths = [
        "data/mssbench/combined.json",
        "../../data/mssbench/combined.json",
        "/home/tim/projects/vlm-agentic-attack/data/mssbench/combined.json"
    ]
    
    input_file = None
    for path in possible_paths:
        if os.path.exists(path):
            input_file = path
            break
            
    if input_file:
        # Create output path in the same directory as input file
        output_dir = os.path.dirname(input_file)
        output_file = os.path.join(output_dir, "combined-subset.json")
        
        get_subset(input_file, output_file)
    else:
        print("Error: combined.json not found in expected locations.")
