# mails

## 1. mail inviata alle professoresse

Salve prof.sse Pastor e Poeta,

Vi scriviamo per condividervi la direzione che io e il mio collega Luigi Marguglio (in cc) intendiamo intraprendere per lo sviluppo del progetto di XAI (P1) per l'appello di settembre. 

Dopo la revisione della letteratura, abbiamo identificato un research gap nello stato dell'arte dei CBM: nonostante il ragionamento umano sia intrinsecamente gerarchico, la maggior parte dei modelli applicati utilizza ancora rappresentazioni di concetti "piatte". Anche le recenti architetture gerarchiche (es. HIL-CBM o CF-CBM) si limitano spesso ad applicare penalità "soft" durante l'addestramento o a strutturare solo lo spazio delle feature, fallendo nel propagare in modo rigido questa gerarchia allo spazio delle predizioni finali. Di fatto, l'ultimo strato decisionale rimane una black-box capace di mischiare concetti illogici.

Per risolvere questo problema, la nostra intenzione è di implementare una nuova architettura CBM con meccanismo di predizione "Coarse-to-Fine" vincolato tramite Hard Gating. A differenza dello stato dell'arte, il nostro approccio rende gerarchico non solo lo spazio dei concetti, ma il ragionamento (routing) stesso del modello, garantendo una coerenza logica assoluta "by design".

Il funzionamento previsto si divide in due stadi:

1. Livello Coarse (Superclasse): Il modello valuta l'immagine basandosi su concetti visivi generali per predire una macro-categoria (es. Veicolo o Animale).
2. Gating Dinamico e Livello Fine (Subclasse): La superclasse predetta agirà da "cancello". Maschererà dinamicamente tutti i concetti di basso livello che sono logicamente incompatibili con la macro-categoria scelta (es. se predice Veicolo, il concetto pelo verrà azzerato). La classificazione finale dipenderà esclusivamente dai concetti di basso livello "sopravvissuti".

A livello metodologico, per evitare il collo di bottiglia delle annotazioni umane, adotteremo il paradigma Label-Free CBM. Utilizzeremo un LLM (es. GPT-4) nella fase di setup per generare i dizionari testuali dei concetti sia per le superclassi che per le subclassi, che verranno poi trasformati in embeddings statici. A tempo di inferenza, utilizzeremo la cosine similarity con le feature estratte da un backbone vision-language (es. CLIP) per attivare i concetti.

Per quanto riguarda i dati, abbiamo scelto di utilizzare il dataset CIFAR-100. Sebbene sia un dataset a bassa risoluzione (il che consentirà iterazioni ed estrazioni delle feature estremamente veloci), possiede nativamente una tassonomia a due livelli (20 superclassi e 100 subclassi) perfettamente bilanciata, ideale per testare e dimostrare l'efficacia del meccanismo di gating senza incorrere in problemi di sbilanciamento delle classi. Abbiamo già scaricato i file originali e ultimato gli script per il caricamento dei dati.

Per valutare l'efficacia del modello, abbiamo intenzione di mettere a confronto le nostre performance con un LF-CBM "piatto" (standard) e un classificatore classico. Nello specifico, la validazione si concentrerà su tre fronti per verificare i seguenti risultati attesi:

- Interpretabilità e sparsità: ci aspettiamo che l'architettura con gating possa ridurre drasticamente i concetti attivi (migliorando la leggibilità della spiegazione) pur mantenendo un'ottima accuratezza.
- Hierarchy-Aware Mistake Evaluation: utilizzando metriche come la Hierarchical Cross-Entropy (HXE) o la Lowest Common Ancestor (LCA) distance. La nostra ipotesi è che il modello compia errori semanticamente "meno gravi" (es. confondere una bicicletta con una moto, invece che con un pesce).
- Intervenability: simulando l'intervento di un esperto che corregge un'eventuale predizione errata al livello coarse. Ipotizziamo che la conseguente correzione del gating mask provochi un sostanziale miglioramento a cascata sulla precisione del livello fine.

In allegato trovate un primo draft dell’architettura.

Saremmo felici di sapere se approvate questa direzione e se avete suggerimenti per perfezionare il design dell'esperimento.

Cordiali saluti,

Francesco Pio De Dominicis

## 2. Risposta Professoressa

Caro Francesco,

La proposta mi sembra in generale buona, sia a livello di research gap, proposta e esperimenti.

I miei dubbi/punti di discussione principali sono i seguenti:
Gerarchia: nella mail fate riferimento alla tassonomia a due livelli di CIFAR: ma quindi la gerarchia e’ a livello di concetti o a livello di classe? 
CIFAR-100: concord che sia a bassa risoluzione e quindi potrebbe limitarvi nell’utilizzo di label-free (potete fare riferimento anche a questo nostro tutorial per l’uso di label free https://github.com/cxai-mechint-htutorial-kdd2025/cxai-mechint-htutorial-kdd2025.github.io/blob/main/notebooks/02b_label_free_cbm.ipynb)
Valutazione/metriche: le hierarchy aware eval fanno riferimento alla gerarchia a livello di classe.  Confermate che quindi considerate la gerarchia sia a livello di concetti che di classe? (Vedete anche il primo punto). Sembra cosi’ dall'architettura

Vi chiederei quindi di charirmi/confermarmi tali punti.

Infine, vi ricordo che parte della valutazione e’ anche nella literature review. Dalle metriche che avete citato mi sembra abbiate gia’ studiato alcuni lavori, vi ribadisco quindi l’importanza di analizzarli anche nel report/presentazione.

A presto,
Eliana Pastor

## 3. Risposta Nostra

Gentile professoressa,

La ringraziamo per il feedback e le dritte. Abbiamo analizzato il notebook che ci ha fornito e crediamo che ci tornerà decisamente utile per completare con successo la fase offline della nostra architettura. 

Per rispondere in maniera puntuale ai suoi dubbi:

- confermiamo che la gerarchia è sia a livello di classe che livello di concetto. Useremo la tassonomia delle classi (20 superclassi e 100 subclassi) come base per costruire due livelli di concetti testuali separati: concetti Coarse per le superclassi e Fine per le subclassi. La predizione della superclasse azzererà dinamicamente i concetti di basso livello incompatibili.

- proprio come mostrato nel tutorial che ci ha condiviso, abbiamo intenzione di operare un upsampling alle immagini di CIFAR-100 così da renderle compatibili con CLIP e sbloccarci nell’utilizzo del LF.

- sì, ci aspettiamo che valutare le performance a livello di classe con metriche hierarchy-aware ci permetta di verificare l'efficacia del nostro gating sui concetti: ci aspettiamo infatti che se il modello sbaglia, l'errore rimanga confinato all'interno della superclasse corretta.

La ringraziamo nuovamente per la disponibilità e rimaniamo assolutamente aperti a ulteriori consigli e/o chiarimenti. 

Cordialmente 

Francesco Pio De Dominicis 

## 4. Risposta Professoressa

Caro Francesco,

Ok per i chiarimenti, mi torna. Vi invito a esplicitare bene nel report/discussione che la gerarchia e’ sia a livello di classe che di concetto.

Resto a disposizione per ulteriori chiarimenti.

A presto,
Eliana Pastor