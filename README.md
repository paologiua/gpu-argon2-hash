# Argon 2 GPU Implementation
Il miglior algoritmo di hash per la generazione e la verifica di password, ottimizzato per l'implementazione su GPU, è Argon2 nella sua variante Argon2id. Argon2id combina i punti di forza delle due varianti di Argon2, ossia Argon2i (resistente agli attacchi di canale laterale) e Argon2d (resistente agli attacchi di forza bruta). Inoltre, è stato progettato per sfruttare sia la CPU che la GPU per calcoli paralleli, rendendolo altamente efficiente su hardware moderni.

Perché Argon2id è adatto alla GPU:
- Parallelismo: Argon2id è progettato per essere parallelizzabile, quindi può sfruttare le capacità di calcolo in parallelo delle GPU.
- Regolabilità: Argon2id consente di configurare tre parametri fondamentali (tempo, memoria e parallelismo) per bilanciare sicurezza e prestazioni, rendendolo adattabile a varie architetture hardware.
- Resistenza agli attacchi: Argon2id offre una protezione robusta contro gli attacchi di brute force, grazie alla sua capacità di richiedere elevati requisiti di memoria (memory-hard) e calcoli intensivi.

### Useful links
- [Argon 2 GPU Paper](https://www.mdpi.com/2076-3417/13/16/9295)  ([pdf](./papers/Argon2%20GPU%20Implementation.pdf))
- [Argon 2 GPU Paper [only pdf]](./papers/Argon2ESP.pdf)
- [Argon2id Hashing Algorithm Medium Article](https://medium.com/@krishanu-ghosh/what-is-argon2-a88000c8caf9)
- [Argon 2 GPU Medium Article](https://medium.com/asecuritysite-when-bob-met-alice/gpu-bursting-password-and-key-derivation-argon2-4b047cfb0ee8)
- [Cuda Doc](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Argon 2 Paper](https://github.com/P-H-C/phc-winner-argon2/blob/master/argon2-specs.pdf)
- [Blake2b](https://www.blake2.net/)