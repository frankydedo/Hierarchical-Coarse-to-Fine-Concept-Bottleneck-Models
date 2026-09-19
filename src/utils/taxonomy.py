# src/taxonomy.py

# Custom High-Res Mini-Dataset Taxonomy (100 classes - full CIFAR-100)
cifar100_taxonomy = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet pepper"],
    "household_electrical_devices": ["clock", "computer keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple tree", "oak tree", "palm tree", "pine tree", "willow tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup truck", "train"],
    "vehicles_2": ["lawn mower", "rocket", "streetcar", "tank", "tractor"]
}

def get_superclasses():
    return list(cifar100_taxonomy.keys())

def get_fine_classes():
    fine_classes = []
    for sc in cifar100_taxonomy.values():
        fine_classes.extend(sc)
    return fine_classes
