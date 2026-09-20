# XAI Project Implementation Plan: Hierarchical Coarse-to-Fine LF-CBM

## 1. Premessa Teorica e Research Gap
In linea con quanto concordato via mail con la Prof.ssa Pastor, questo progetto mira a risolvere i limiti dei classici Concept Bottleneck Models (CBM). Attualmente, lo stato dell'arte si basa su rappresentazioni "piatte" o applica penalità gerarchiche solo in fase di addestramento (soft penalty), lasciando l'ultimo strato come una black-box incline al *concept bleeding* (mescolare concetti illogici).

La nostra architettura introduce un **Coarse-to-Fine Hard Gating**.
**CRITICITÀ CONCORDATA:** La gerarchia è applicata **sia a livello di classe che a livello di concetto**. 
1. **Gerarchia di Classe:** Sfruttiamo la tassonomia nativa a due livelli di CIFAR-100.
2. **Gerarchia di Concetto:** Creeremo due dizionari separati. I "Concetti Coarse" (per discriminare le macro-categorie) e i "Concetti Fine" (per discriminare le sotto-categorie).

---

## 2. Il Dataset: Custom High-Res CIFAR-100 Taxonomy
Non utilizzeremo le immagini originali del dataset CIFAR-100 in quanto la loro risoluzione nativa (32x32 pixel) è troppo bassa e un semplice upsampling genererebbe immagini eccessivamente sgranate, penalizzando l'estrattore CLIP. Pertanto, **abbiamo creato un nuovo dataset ad alta risoluzione che eredita esclusivamente l'esatta tassonomia a due livelli di CIFAR-100**, mantenendo così il perfetto bilanciamento per i nostri esperimenti (20 Superclassi, 100 Subclassi):
1. **aquatic_mammals:** beaver, dolphin, otter, seal, whale
2. **fish:** aquarium fish, flatfish, ray, shark, trout
3. **flowers:** orchid, poppy, rose, sunflower, tulip
4. **food_containers:** bottle, bowl, can, cup, plate
5. **fruit_and_vegetables:** apple, mushroom, orange, pear, sweet pepper
6. **household_electrical_devices:** clock, computer keyboard, lamp, telephone, television
7. **household_furniture:** bed, chair, couch, table, wardrobe
8. **insects:** bee, beetle, butterfly, caterpillar, cockroach
9. **large_carnivores:** bear, leopard, lion, tiger, wolf
10. **large_man_made_outdoor_things:** bridge, castle, house, road, skyscraper
11. **large_natural_outdoor_scenes:** cloud, forest, mountain, plain, sea
12. **large_omnivores_and_herbivores:** camel, cattle, chimpanzee, elephant, kangaroo
13. **medium_mammals:** fox, porcupine, possum, raccoon, skunk
14. **non_insect_invertebrates:** crab, lobster, snail, spider, worm
15. **people:** baby, boy, girl, man, woman
16. **reptiles:** crocodile, dinosaur, lizard, snake, turtle
17. **small_mammals:** hamster, mouse, rabbit, shrew, squirrel
18. **trees:** maple, oak, palm, pine, willow
19. **vehicles_1:** bicycle, bus, motorcycle, pickup truck, train
20. **vehicles_2:** lawn-mower, rocket, streetcar, tank, tractor

---

## 3. Execution Pipeline

### Phase 1: High-Res Dataset Creation
- **Scraping Custom Dataset:** Invece di fare upsampling sulle immagini 32x32 originali (come ipotizzato inizialmente nelle mail), abbiamo sviluppato uno script custom per scaricare direttamente 6.000 immagini in alta risoluzione (50 train, 10 test per ognuna delle 100 classi). Questo garantisce che il vision encoder di CLIP possa estrarre feature di altissima qualità, essenziale per il successo del paradigma Label-Free.

### Phase 2: Label-Free Concept Generation
- **Generazione:** Utilizzare un LLM (es. Gemma 4) per generare i dizionari testuali *separati* per concetti Coarse e Fine.
- **Filtraggio:** Proiettare i concetti nello spazio CLIP e usare la cosine similarity con l'Hard-Margin Filter per trattenere solo i concetti visivamente presenti.

### Phase 3: Architettura Hierarchical Hard-Gating
- **Livello Coarse (Superclasse):** Una regressione logistica valuta l'immagine usando *solo* i concetti Coarse per predire una delle 20 superclassi.
- **Hard Gating (Il Cancello):** La superclasse predetta azzera dinamicamente tutti i concetti Fine che sono logicamente incompatibili con la macro-categoria scelta.
- **Livello Fine (Subclasse):** La classificazione finale dipenderà esclusivamente dai concetti Fine "sopravvissuti" al mascheramento.

### Phase 4: Advanced Evaluation & Benchmarking (Validazione)
Come promesso in fase di proposta, valuteremo le performance mettendole a confronto con due baseline: un **LF-CBM Piatto** (per ablare il gating) e un **Linear Probe CLIP** (per l'upper-bound black-box).

La validazione si baserà esattamente su questi tre pilastri:
1. **Interpretabilità e sparsità:** Dimostrare che il gating riduce drasticamente i concetti attivi rispetto alla baseline piatta, fornendo spiegazioni più leggibili.
2. **Hierarchy-Aware Mistake Evaluation:** Calcolare metriche come la Hierarchical Cross-Entropy (HXE) o la Lowest Common Ancestor (LCA) distance sugli alberi tassonomici, verificando che gli errori rimangano confinati nella superclasse corretta (es. confondere due tipi di veicoli e non un veicolo con un pesce).
3. **Intervenability (Test Oracolo):** Simulare l'intervento di un esperto che corregge la predizione errata al Livello Coarse, dimostrando il miglioramento a cascata sull'accuratezza del Livello Fine.
