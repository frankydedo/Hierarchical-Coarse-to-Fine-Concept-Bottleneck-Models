import os
import json
import time
from openai import OpenAI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.taxonomy import cifar100_taxonomy

# Find project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Initialize OpenAI client pointing to local LM Studio server
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio" # LM Studio doesn't require a real API key
)
MODEL_ID = "gemma-4-12b-it"

def call_with_retry(prompt, retries=3, backoff=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are an expert computer vision annotator. Your descriptions must focus EXCLUSIVELY on physical, visual features (shapes, colors, textures, distinct body parts). NEVER mention behaviors, habitats, sounds, or invisible traits. Keep phrases extremely concise (maximum 4 words). You only output valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                frequency_penalty=1.2
            )
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown if the model still outputs it
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            return json.loads(content.strip())
            
        except Exception as e:
            print(f"Parsing or API error (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(backoff)
            
    return []

def generate_coarse_concepts(superclasses, num_concepts=15):
    output_file = os.path.join(DATA_DIR, "raw_coarse_concepts.json")
    coarse_concepts = {}
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            coarse_concepts = json.load(f)
            
    for sc in superclasses:
        if sc in coarse_concepts and len(coarse_concepts[sc]) > 0:
            print(f"Skipping {sc}, already generated.")
            continue
            
        print(f"Generating concepts for superclass: {sc}")
        prompt = (
            f"List {num_concepts} highly distinct visual, physical, "
            f"or structural features that are most important for recognizing an object belonging to the general category of '{sc}'. "
            f"The features should be short phrases (maximum 4-5 words). "
            f"Output ONLY a JSON list of strings."
        )
        coarse_concepts[sc] = call_with_retry(prompt)
        
        # Save incrementally
        with open(output_file, "w") as f:
            json.dump(coarse_concepts, f, indent=4)
            
    return coarse_concepts

def generate_fine_concepts(taxonomy, num_concepts=15):
    output_file = os.path.join(DATA_DIR, "raw_fine_concepts.json")
    fine_concepts = {}
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            fine_concepts = json.load(f)
            
    for sc, fine_list in taxonomy.items():
        for fc in fine_list:
            if fc in fine_concepts and len(fine_concepts[fc]) > 0:
                print(f"Skipping {fc}, already generated.")
                continue
                
            print(f"Generating concepts for subclass: {fc} (sibling of {', '.join([s for s in fine_list if s != fc])})")
            siblings = [sibling for sibling in fine_list if sibling != fc]
            siblings_str = ", ".join(siblings)
            prompt = (
                f"List {num_concepts} highly specific visual features for recognizing a '{fc}'. "
                f"Focus strictly on visual details that differentiate this object from its siblings in the same '{sc}' category, "
                f"which are: {siblings_str}. Do not include generic features shared by the siblings. "
                f"The features should be short phrases (maximum 4-5 words). "
                f"Output ONLY a JSON list of strings."
            )
            fine_concepts[fc] = call_with_retry(prompt)
            
            # Save incrementally
            with open(output_file, "w") as f:
                json.dump(fine_concepts, f, indent=4)
                
    return fine_concepts

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=== Phase 1: Generating Coarse Concepts ===")
    generate_coarse_concepts(cifar100_taxonomy.keys())
    print("Finished Coarse Concepts\n")
    
    print("=== Phase 2: Generating Fine Concepts ===")
    generate_fine_concepts(cifar100_taxonomy)
    print("Finished Fine Concepts")
