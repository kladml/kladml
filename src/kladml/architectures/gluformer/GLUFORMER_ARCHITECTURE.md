# 🧬 Gluformer: Architettura per Glucose Forecasting Probabilistico

## Overview

**Gluformer** è un modello **Transformer-based** progettato per la previsione probabilistica del glucosio ematico da sensori CGM (Continuous Glucose Monitoring). A differenza dei modelli deterministici tradizionali, Gluformer produce una **distribuzione gaussiana** per ogni punto predetto, fornendo sia la previsione che la sua incertezza.

---

## Riferimenti Originali

> **Paper**: *"Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification"*  
> **Autori**: Renat Sergazinov, Mohammadreza Armandpour, Irina Gaynanova  
> **Affiliazione**: University of Michigan  
> **arXiv**: [2209.04526](https://arxiv.org/abs/2209.04526) (Settembre 2022)  
> **GitHub**: [https://github.com/IrinaStatsLab/Gluformer](https://github.com/IrinaStatsLab/Gluformer)

Il dataset **REPLACE-BG** utilizzato per il benchmarking proviene da uno studio clinico condotto presso l'Università del Michigan.

---

## Architettura

```
┌────────────────────────────────────────────────────────────┐
│                    GLUFORMER ARCHITECTURE                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐         ┌──────────────┐                  │
│  │  Past CGM   │         │  Start Token │                  │
│  │  (60 steps) │         │  + Zeros     │                  │
│  └──────┬──────┘         └──────┬───────┘                  │
│         ▼                       ▼                          │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │   Encoder   │         │   Decoder   │                   │
│  │  Embedding  │         │  Embedding  │                   │
│  └──────┬──────┘         └──────┬──────┘                   │
│         ▼                       ▼                          │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │  Encoder    │◄───────►│  Decoder    │                   │
│  │  (3 layers) │  cross  │  (2 layers) │                   │
│  │             │  attn   │             │                   │
│  └─────────────┘         └──────┬──────┘                   │
│                                 │                          │
│                    ┌────────────┴───────────┐              │
│                    ▼                        ▼              │
│             ┌───────────┐           ┌───────────┐          │
│             │ Mean Head │           │ Var Head  │          │
│             │  (Linear) │           │   (MLP)   │          │
│             └─────┬─────┘           └─────┬─────┘          │
│                   ▼                       ▼                │
│              μ (pred)               log(σ²) (uncertainty)  │
└────────────────────────────────────────────────────────────┘
```

---

## Componenti Chiave

### 1. Data Embedding
- **Positional Encoding**: Sinusoidal encoding per catturare la posizione temporale
- **Value Embedding**: Proiezione lineare dei valori CGM in spazio d_model
- **Static Embedding**: ID paziente per personalizzazione (opzionale)

### 2. Encoder Stack (3 layers)
- **Multi-Head Self-Attention** (8 heads): Cattura dipendenze temporali
- **Feedforward Network**: Proiezione non-lineare (GELU activation)
- **LayerNorm + Residual**: Stabilità del training

L'encoder elabora la sequenza passata (5 ore) per estrarre pattern di inerzia glicemica, trend, e variabilità.

### 3. Decoder Stack (2 layers)
- **Masked Self-Attention**: Previene data leakage durante generazione
- **Cross-Attention**: Attende all'output dell'encoder per condizionare le previsioni
- **Start Token**: Ultimi 48 punti del passato come contesto iniziale

### 4. Dual Output Heads (Probabilistic Output)

| Head | Tipo | Output | Scopo |
|------|------|--------|-------|
| **Mean** | Linear | μ ∈ ℝ | Valore predetto |
| **Variance** | MLP (2-layer + GELU) | log(σ²) ∈ ℝ | Incertezza |

*Perché MLP per la varianza?* La stima dell'incertezza ha dinamiche non-lineari più complesse rispetto alla media.

---

## Training Strategy

### Loss Function: Gaussian Negative Log-Likelihood

```
L_NLL = 0.5 * [ exp(-log(σ²)) * (y - μ)² + log(σ²) ]
```

**Proprietà**:
- Penalizza sia errori di previsione che incertezza mal calibrata
- Se σ troppo grande → termine `log(σ²)` penalizza
- Se σ troppo piccolo ma errore alto → termine esponenziale esplode

### Warmup Strategy (Anti-Collapse)

Il training avviene in due fasi per evitare il **"variance collapse"** - un problema dove il modello impara a "barare" predicendo sempre la media globale (~150 mg/dL) con incertezza massima.

| Fase | Epoche | Loss | Obiettivo |
|------|--------|------|-----------|
| **Warmup** | 0-4 | MSE | Impara a predire le **traiettorie reali** di ogni paziente |
| **Fine-tuning** | 5+ | NLL | Calibra l'**incertezza** su quelle previsioni |

#### Perché serve il Warmup?

**Problema senza warmup:**
```
Input:  [120, 125, 130, 140, 150]  (paziente in salita verso iperglicemia)
Target: [160, 175, 190, 200, 195]

❌ Modello pigro: predice sempre [150, 150, 150, 150, 150] con σ=999
   → Loss NLL bassa perché "non sa niente" (incertezza massima copre tutto)
```

**Con warmup MSE:**
```
❌ MSE([150,150,150,150,150], [160,175,190,200,195]) = ALTO!
✅ Il modello è FORZATO a imparare la traiettoria reale
```

#### Fasi nel dettaglio

1. **Warmup (MSE)**: Il modello impara a seguire le traiettorie individuali
   - Cattura pattern di salita/discesa
   - Impara a riconoscere ipo/iperglicemie imminenti
   - Non può barare con "sempre la media"

2. **Fine-tuning (NLL)**: Una volta che sa predire bene, aggiunge la calibrazione dell'incertezza
   - Quando è sicuro → σ piccolo (cono stretto)
   - Quando è incerto → σ grande (cono largo)
   - Incertezza crescente nel tempo (effetto "cono")

---

## Configurazione Netcaring

### Specifiche Tecniche - Foundation Model v1

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| **Input window** | 60 samples (5h) | Cattura cicli glicemici completi |
| **Output horizon** | 12 samples (1h) | Tempo utile per intervento clinico |
| **d_model** | 512 | Trade-off capacità/efficienza |
| **Attention heads** | 8 | Cattura pattern multipli simultanei |
| **Encoder layers** | 3 | Sufficiente per pattern temporali CGM |
| **Decoder layers** | 2 | Decoding più semplice dell'encoding |
| **Feedforward dim** | 2048 | Standard 4x d_model |
| **Dropout** | 0.05 | Regularizzazione leggera |
| **Parametri totali** | **~13M** | Edge-compatible |

### Training Configuration

| Parametro | Valore |
|-----------|--------|
| Epochs | 50 (max) |
| Early stopping | patience=8 |
| Warmup epochs | 5 |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Optimizer | Adam |

### Dataset

| Split | Serie | Samples | Note |
|-------|-------|---------|------|
| Train | 6,317 | ~516K | CGM University of Michigan |
| Validation | 603 | ~52K | |
| Test | ~600 | ~50K | |

---

## Output del Modello

### Previsione Probabilistica

```python
{
    "forecast": [120.5, 118.2, 115.8, ...],      # μ: Valori previsti (mg/dL)
    "forecast_std": [3.2, 5.1, 7.8, ...],       # σ: Deviazione standard
    "risk_assessment": "normal",                 # Classificazione rischio
    "forecast_horizon_minutes": 60               # Orizzonte temporale
}
```

### Visualizzazione: Cono di Incertezza

L'incertezza cresce nel tempo, creando un "cono" di confidenza:
- **t+5min**: σ piccolo → alta confidenza
- **t+60min**: σ grande → incertezza maggiore

Questo permette decisioni cliniche informate: intervenire solo quando la previsione + incertezza indica rischio.

---

## Vantaggi Clinici

1. **Incertezza Quantificata**: Il cono di confidenza permette decisioni informate
2. **Anticipazione Eventi**: Fino a 1 ora di anticipo su ipo/iperglicemia
3. **Personalizzazione**: Embedding paziente per adattamento individuale
4. **Edge-Ready**: Inferenza <2s anche su Raspberry Pi 4
5. **Intervallo CGM**: Compatibile con campionamento ogni 5 minuti

---

## Riferimenti Bibliografici

1. Sergazinov, R., Armandpour, M., & Gaynanova, I. (2022). *"Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification"*. arXiv:2209.04526

2. Vaswani, A. et al. (2017). *"Attention Is All You Need"*. NeurIPS.

3. Wu, H. et al. (2021). *"Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"*. NeurIPS.

4. Zhou, H. et al. (2021). *"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"*. AAAI.

---

*Implementazione Netcaring basata sul lavoro originale dell'Università del Michigan, adattata per il progetto Sentinella.*

---

## Deployment & Visualizzazione (Frontend)

Quando esporti il modello per il deployment (es. `best_model_jit.pt`), devi gestire correttamente l'output probabilistico nel frontend.

### 1. Output del Modello (JIT)

Il modello esportato restituisce due tensori:

1.  **`pred_mean`**: La previsione puntuale del glucosio.
2.  **`pred_logvar`**: La **Log-Varianza** dell'errore (Incertezza Aleatoria).

### 2. Calcolo Intervallo di Confidenza

Per visualizzare la "nuvola" di incertezza, converti la log-varianza in deviazione standard ($\sigma$) e calcola l'intervallo al 95%:

$$ \sigma = \sqrt{\exp(\text{logvar})} $$
$$ CI_{95\%} = [\text{mean} - 1.96\sigma, \text{mean} + 1.96\sigma] $$

### 3. Esempio Implementazione (React + Recharts)

Ecco come visualizzare i dati in un'applicazione React:

#### Data Processing (JavaScript)

```javascript
const processPredictions = (means, logvars) => {
  return means.map((mean, i) => {
    const logvar = logvars[i];
    const sigma = Math.sqrt(Math.exp(logvar));
    
    // Time: ogni step è +5 minuti
    const timestamp = Date.now() + i * 5 * 60 * 1000;
    
    return {
      timestamp,
      glucose: mean,
      // Range per Area chart: [Low, High]
      range: [mean - 1.96 * sigma, mean + 1.96 * sigma]
    };
  });
};
```

#### Componente Grafico

```jsx
import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip } from 'recharts';

const GlucoseChart = ({ data }) => (
  <ComposedChart width={600} height={300} data={data}>
    <XAxis 
      dataKey="timestamp" 
      tickFormatter={(t) => new Date(t).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} 
    />
    <YAxis domain={['auto', 'auto']} unit=" mg/dL" />
    <Tooltip />
    
    {/* Nuvola di Incertezza (Grigia/Viola chiaro) */}
    <Area 
      type="monotone" 
      dataKey="range" 
      stroke="none"
      fill="#8884d8" 
      opacity={0.2} 
    />
    
    {/* Linea di Previsione (Solida) */}
    <Line 
      type="monotone" 
      dataKey="glucose" 
      stroke="#8884d8" 
      strokeWidth={3} 
      dot={false} 
    />
  </ComposedChart>
);
```
