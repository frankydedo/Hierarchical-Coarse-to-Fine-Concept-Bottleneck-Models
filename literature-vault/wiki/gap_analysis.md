# Analisi del Research Gap: CBM Gerarchici e Dynamic Gating

È un'osservazione eccellente. La letteratura recente abbonda effettivamente di modelli *hierarchy-aware* o *Concept Bottleneck Models gerarchici*. Tuttavia, il **research gap non risiede nell'assenza di gerarchie nello spazio dei concetti, ma nella mancanza di un meccanismo di vincolo rigido (gating) tra le predizioni di alto livello e quelle di basso livello.**

Di seguito un'analisi strutturata basata sulla nostra `literature-vault`.

---

## 1. I Paper più vicini alla nostra direzione

La revisione della letteratura ha evidenziato alcuni lavori che si avvicinano molto alla nostra idea, ma che lasciano scoperto un gap fondamentale:

*   **Coarse-to-Fine Concept Bottleneck Models (CF-CBMs) [Panousis_2023]**
    *   *Cosa fanno:* Utilizzano un "Concept Discovery Block" per mascherare gerarchicamente attributi irrilevanti a livello di patch (porzioni di immagine).
    *   *Cosa manca:* Il masking avviene a livello visivo/feature, ma alla fine il modello predice una singola *label piatta*. Non c'è una predizione intermedia della superclasse che vincola la predizione finale.
*   **HIL-CBM [Xie_2026]**
    *   *Cosa fanno:* Estendono il paradigma Label-Free alle gerarchie, predicendo concetti a più livelli di astrazione e usando perdite matematiche (Tree-path KL Divergence Loss) per mantenere la coerenza.
    *   *Cosa manca:* Usano "soft constraints" (penalità durante il training). Il modello, a tempo di inferenza, non è *costretto* a essere coerente. Può ancora matematicamente predire un gatto usando il concetto "ruote", anche se il loss lo ha scoraggiato.
*   **SUPCBM / Hi-CBM [Sun_2024, Sun_2026]**
    *   *Cosa fanno:* Separano la predizione di concetti e label tramite matrici di intervento e "Concept Pooling" per filtrare concetti spuri.
    *   *Cosa manca:* Si concentrano sulla lotta al *concept leakage* (passaggio di informazioni nascoste) e richiedono architetture complesse per fare "pooling".

## 2. Esiste davvero il Research Gap?

**Sì.** La stragrande maggioranza dei CBM gerarchici (inclusi quelli sopra citati) soffre del **Disconnect Between Concept Hierarchies and Label Hierarchies**. 

Il problema è il seguente: questi modelli imparano una bellissima struttura ad albero per i concetti visivi, ma alla fine "buttano" tutti i valori di attivazione di questi concetti dentro a un ultimo strato lineare (Fully Connected Layer) per predire la classe finale. 
Questo significa che il livello decisionale finale rimane una "black box" matematica che può mischiare liberamente concetti incompatibili. La gerarchia viene usata per *imparare* i concetti, ma non per *vincolare logicamente* le predizioni finali.

## 3. Perché la nostra ricerca (Path B) è valida e innovativa

Il nostro progetto (Architettura Coarse-to-Fine con Gating Dinamico) è valido per colmare questo gap perché sposta la gerarchia dallo *spazio delle features* allo **spazio delle decisioni (Label Space)** in modo **rigido**.

Ecco cosa ci distingue dallo stato dell'arte:

1.  **Hard Gating vs. Soft Penalties:** Invece di sperare che la rete neurale impari a non usare i concetti dei pesci per classificare le automobili (come fa *HIL-CBM*), noi implementiamo un **Dynamic Masking**. Se il Livello 1 predice la superclasse "Veicolo", i concetti associati agli animali vengono istantaneamente e matematicamente azzerati (moltiplicati per 0). Questo garantisce il 100% di coerenza logica *by design*.
2.  **Sequential Bottleneck:** Il modello non fa tutto in un colpo solo. Risolve prima il problema facile (Coarse: Veicolo vs Animale) e usa quella soluzione come **prerequisito assoluto** per risolvere il problema difficile (Fine: SUV vs Berlina).
3.  **Label-Free puro:** Combiniamo questo meccanismo rigido con l'estrazione dei concetti tramite LLM (Label-Free), dimostrando che possiamo ottenere interpretazioni logiche infallibili (a livello di architettura) senza aver mai bisogno di dataset annotati manualmente.

**In sintesi:** La letteratura ha reso gerarchici i *concetti*. Noi stiamo rendendo gerarchico il *ragionamento (routing) del modello*.
