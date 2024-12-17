#include <stdio.h>

// Esta é a declaração do kernel CUDA. 
// O prefixo __global__ indica que esta função será executada na GPU.
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 1000;
    int *a, *b, *c; // Vetores na CPU
    int *d_a, *d_b, *d_c; // Vetores na GPU
    int size = n * sizeof(int);

    // Aloca memória para vetores na GPU
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Aloca memória para vetores na CPU
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Inicializa vetores na CPU
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i;
    }

    // Copia vetores da CPU para a GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Executa o kernel CUDA
    int blockSize = 256; // tamanho do bloco como 256 threads
    int numBlocks = (n + blockSize - 1) / blockSize; // calcula o número de blocos
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Copia o resultado de volta da GPU para a CPU
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Imprime os resultados
    for (int i = 0; i < 15; ++i) 
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    // Libera memória na GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Libera memória na CPU
    free(a);
    free(b);
    free(c);

    return 0;
}
