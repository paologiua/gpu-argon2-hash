#include <cuda_runtime.h>
#include <iostream>

// Kernel CUDA: esempio semplice
__global__ void simple_hash_kernel(uint8_t* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Operazione fittizia di hashing
        data[idx] ^= 0x5A;  // XOR per esempio
    }
}

int main() {
    const int dataSize = 1024;
    uint8_t hostData[dataSize];

    // Inizializza i dati
    for (int i = 0; i < dataSize; i++) {
        hostData[i] = i % 256;
    }

    uint8_t* deviceData;

    // Allocazione memoria su GPU
    cudaMalloc((void**)&deviceData, dataSize * sizeof(uint8_t));

    // Copia dati da CPU a GPU
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Esegui kernel con 256 thread per blocco
    simple_hash_kernel<<<(dataSize + 255) / 256, 256>>>(deviceData, dataSize);

    // Copia i dati processati dalla GPU alla CPU
    cudaMemcpy(hostData, deviceData, dataSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Libera memoria
    cudaFree(deviceData);

    // Stampa risultato
    for (int i = 0; i < 10; i++) {
        std::cout << static_cast<int>(hostData[i]) << " ";
    }

    return 0;
}
