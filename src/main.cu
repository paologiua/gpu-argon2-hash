#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>

// Definizione delle costanti per Argon2 semplificato
constexpr int BLOCK_SIZE = 1024; // Dimensione del blocco in byte
constexpr int NUM_BLOCKS = 256;  // Numero di blocchi di memoria
constexpr int NUM_ITERATIONS = 3; // Numero di iterazioni

// Kernel CUDA per l'iterazione del mixing dei blocchi
__global__ void argon2_kernel(uint8_t* memory, int num_blocks, int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_blocks) {
        // Operazione fittizia di mixing (esempio semplificato)
        uint8_t* block = memory + idx * block_size;
        for (int i = 0; i < block_size; ++i) {
            block[i] ^= 0x5A;  // XOR per esempio
        }
    }
}

// Funzione host per inizializzare la memoria con dati casuali
void initialize_memory(uint8_t* memory, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < size; ++i) {
        memory[i] = static_cast<uint8_t>(dis(gen));
    }
}

int main() {
    // Calcola la dimensione della memoria necessaria
    size_t memory_size = BLOCK_SIZE * NUM_BLOCKS;
    
    // Allocazione memoria sulla CPU
    std::vector<uint8_t> host_memory(memory_size);
    
    // Inizializza la memoria con dati casuali
    initialize_memory(host_memory.data(), memory_size);

    // Puntatore alla memoria sulla GPU
    uint8_t* device_memory;

    // Allocazione memoria sulla GPU
    cudaMalloc((void**)&device_memory, memory_size);

    // Copia i dati dalla CPU alla GPU
    cudaMemcpy(device_memory, host_memory.data(), memory_size, cudaMemcpyHostToDevice);

    // Configura la dimensione dei thread e dei blocchi
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_BLOCKS + threadsPerBlock - 1) / threadsPerBlock;

    // Esegui il kernel CUDA per il numero di iterazioni specificato
    for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
        argon2_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_memory, NUM_BLOCKS, BLOCK_SIZE);
        cudaDeviceSynchronize(); // Sincronizza i thread
    }

    // Copia i risultati dalla GPU alla CPU
    cudaMemcpy(host_memory.data(), device_memory, memory_size, cudaMemcpyDeviceToHost);

    // Libera la memoria sulla GPU
    cudaFree(device_memory);

    // Stampa i primi byte dell'output per verifica
    std::cout << "Prime 16 byte della memoria processata: ";
    for (int i = 0; i < 16; ++i) {
        std::cout << static_cast<int>(host_memory[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}
