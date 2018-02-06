# Experiments - building strong baseline

## First experiment - the simplest baseline

See baseline.py for model specification.

* RNNs with LSTM cells
* SGD optimiser with learning rate = 0.5
* Encoder Decoder architecture
* 2 layers in Encoder
* 2 layers in Decoder
* Dropout = 0.3 
* Embedding/cell vector length = 512

BLEU SCORE: 3.02
Comment: very low, rubbish translations. SGD doesn't reach the full potential with convergence.

## Second experiment - the baseline with Adam

See baseline+adam.py for model specification.

* RNNs with LSTM cells
* Adam optimiser with learning rate = 0.001
* Encoder Decoder architecture
* 2 layers in Encoder
* 2 layers in Decoder
* Dropout = 0.3 
* Embedding/cell vector length = 512

BLEU SCORE: 2.62
Comment: lower, but insignificantly at this small score - converges significantly faster, that's why it will be used further.


## Third experiment - the baseline with Adam and attention mechanism Bahandau-style

See baseline+attention.py for model specification.

* RNNs with LSTM cells
* Adam optimiser with learning rate = 0.001
* Encoder Decoder architecture
* 2 layers in Encoder
* 2 layers in Decoder
* Dropout = 0.3 
* Embedding/cell vector length = 512
* Attention mechanism in Decoder (Bahandau)

BLEU SCORE: 11.60
Comment: the previous experiments' results were basically useless. Adding attention mechanism allows model to finally learn efficiently to produce understandable and correct translations. HUGE improvement!

## BONUS (if we manage) Fourth experiment - the baseline with Adam and GRU cells

See baseline+gru.py for model specification.

* RNNs with GRU cells
* Adam optimiser with learning rate = 0.001
* Encoder Decoder architecture
* 2 layers in Encoder
* 2 layers in Decoder
* Dropout = 0.3 
* Embedding/cell vector length = 512

BLEU SCORE: 2.82
Comment: higher, but insignificantly at this small score - is less stable. Won't probably learn as well as we want in the future.
