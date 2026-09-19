import os
import glob
import re
import xml.etree.ElementTree as ET
import urllib.request

RAW_DIR = "/Users/francescodedominicis/Desktop/XAI_P1/literature-vault/raw"

def get_arxiv_metadata(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None, None
        
        # Get first author
        author_elem = entry.find('atom:author/atom:name', ns)
        if author_elem is None:
            return None, None
            
        author = author_elem.text.split()[-1]
        
        # Get year from published
        published = entry.find('atom:published', ns).text
        year = published[:4]
        
        return author, year
    except Exception as e:
        print(f"Error fetching metadata for {arxiv_id}: {e}")
        return None, None

manual_mapping = {
  "Making_better_mistakes_Leveraging_class_hierarchies_with_deep_networks_.pdf": "Bertinetto_2020.pdf",
  "Interpretable_image_recognition_with_hierarchical_prototypes_.pdf": "Hase_2019.pdf",
  "Hierarchical concept bottleneck models for vision and their application.pdf": "Pittino_2023.pdf",
  "Concept_based_Explainable_Artificial_Intelligence_A_Survey_.pdf": "Poeta_2025.pdf",
  "Eliminating_Information_Leakage_in_Hard_Concept_Bottleneck_Models_.pdf": "Sun_2024.pdf",
  "Hyperbolic_Concept_Bottleneck_Models_.pdf": "Uyterlinde_2026.pdf",
  "Towards_Reasonable_Concept_Bottleneck_Models_.pdf": "Kalampalikis_2025.pdf",
  "Explainable_fetal_ultrasound_quality_assessment_with_progressive_concept_bottleneck_models_.pdf": "Lin_2022.pdf",
  "1-s2.0-S0925231226003139-main.pdf": "Sun_2026.pdf" # This is the Hi-CBM paper
}

arxiv_title_map = {
  "Coarse_to_Fine_Concept_Bottleneck_Models_.pdf": "2310.02116",
  "Hierarchical_Prototype_Learning_for_Zero_Shot_Recognition_.pdf": "1910.11671",
  "ProtoEHR_Hierarchical_Prototype_Learning_for_EHR_based_Healthcare_Predictions_.pdf": "2508.18313",
  "HiProto_Hierarchical_Prototype_Learning_for_Interpretable_Object_Detection_Under_Low_quality_Conditions_.pdf": "2604.13981"
}

os.chdir(RAW_DIR)

for file in glob.glob("*.pdf"):
    new_name = None
    
    # Don't rename if already renamed
    if re.match(r'^[A-Z][a-zA-Z]+_\d{4}\.pdf$', file):
        continue
        
    if file in manual_mapping:
        new_name = manual_mapping[file]
        
    elif file in arxiv_title_map:
        author, year = get_arxiv_metadata(arxiv_title_map[file])
        if author and year:
            new_name = f"{author}_{year}.pdf"
            
    else:
        # Check if it's an arxiv ID format
        m = re.match(r'^(\d{4}\.\d{4,5})(v\d+)?\.pdf$', file)
        if m:
            arxiv_id = m.group(1)
            author, year = get_arxiv_metadata(arxiv_id)
            if author and year:
                new_name = f"{author}_{year}.pdf"

    if new_name:
        if file != new_name:
            # Handle duplicate names by appending a, b, c, etc.
            if os.path.exists(new_name):
                base, ext = os.path.splitext(new_name)
                for suffix in ['a', 'b', 'c', 'd', 'e']:
                    alt_name = f"{base}{suffix}{ext}"
                    if not os.path.exists(alt_name):
                        new_name = alt_name
                        break
                        
            print(f"Renaming {file} -> {new_name}")
            os.rename(file, new_name)
    else:
        print(f"Could not determine new name for {file}")

print("Done renaming!")
