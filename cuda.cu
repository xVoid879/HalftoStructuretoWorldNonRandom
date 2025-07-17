#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256
#define STRUCTURE_MULTIPLIER 65536

__global__ void structureToWorldKernel(uint64_t* structureSeeds, uint64_t* worldSeeds, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seedIdx = idx / STRUCTURE_MULTIPLIER;
    int lowerBits = idx % STRUCTURE_MULTIPLIER;

    if (seedIdx < count) {
        worldSeeds[idx] = (structureSeeds[seedIdx] << 16) | lowerBits;
    }
}

void errorExit(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

void riverToHalfSeeds(const char* inputFile, const char* outputFile) {
    FILE* in = fopen(inputFile, "r");
    if (!in) errorExit("Failed to open river seed input file");

    FILE* out = fopen(outputFile, "w");
    if (!out) errorExit("Failed to open output file for half seeds");

    uint32_t riverSeed;
    while (fscanf(in, "%" SCNu32, &riverSeed) == 1) {
        if (riverSeed >= (1 << 26)) {
            fprintf(stderr, "Invalid river seed: %u\n", riverSeed);
            continue;
        }
        for (uint32_t upper = 0; upper < 64; upper++) {
            uint64_t half = ((uint64_t)upper << 26) | riverSeed;
            fprintf(out, "%" PRIu64 "\n", half);
        }
    }

    fclose(in);
    fclose(out);
    printf("Finished converting river to half seeds.\n");
}

void halfToStructureSeeds(const char* inputFile, const char* outputFile) {
    FILE* in = fopen(inputFile, "r");
    if (!in) errorExit("Failed to open half seed input file");

    FILE* out = fopen(outputFile, "w");
    if (!out) errorExit("Failed to open output file for structure seeds");

    uint64_t halfSeed;
    while (fscanf(in, "%" SCNu64, &halfSeed) == 1) {
        uint64_t structureSeed = halfSeed ^ 0x5DEECE66DULL;
        fprintf(out, "%" PRIu64 "\n", structureSeed);
    }

    fclose(in);
    fclose(out);
    printf("Finished converting half to structure seeds.\n");
}

void structureToWorldSeedsGPU(const char* inputFile, const char* outputFile) {
    FILE* in = fopen(inputFile, "r");
    if (!in) errorExit("Error opening input structure_seeds.txt");

    uint64_t* hostStructureSeeds = (uint64_t*)malloc(sizeof(uint64_t) * 500000);
    int count = 0;

    while (fscanf(in, "%" SCNu64, &hostStructureSeeds[count]) == 1) {
        count++;
    }
    fclose(in);
    printf("Read %d structure seeds from %s\n", count, inputFile);

    uint64_t* devStructureSeeds;
    uint64_t* devWorldSeeds;
    size_t totalWorldSeeds = count * STRUCTURE_MULTIPLIER;

    cudaMalloc(&devStructureSeeds, sizeof(uint64_t) * count);
    cudaMalloc(&devWorldSeeds, sizeof(uint64_t) * totalWorldSeeds);

    cudaMemcpy(devStructureSeeds, hostStructureSeeds, sizeof(uint64_t) * count, cudaMemcpyHostToDevice);

    int blocks = (totalWorldSeeds + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching kernel with %d blocks, %d threads per block\n", blocks, THREADS_PER_BLOCK);
    structureToWorldKernel<<<blocks, THREADS_PER_BLOCK>>>(devStructureSeeds, devWorldSeeds, count);
    cudaDeviceSynchronize();

    uint64_t* hostWorldSeeds = (uint64_t*)malloc(sizeof(uint64_t) * totalWorldSeeds);
    cudaMemcpy(hostWorldSeeds, devWorldSeeds, sizeof(uint64_t) * totalWorldSeeds, cudaMemcpyDeviceToHost);

    FILE* out = fopen(outputFile, "w");
    if (!out) errorExit("Failed to open output file for world seeds");
    for (size_t i = 0; i < totalWorldSeeds; i++) {
        fprintf(out, "%" PRIu64 "\n", hostWorldSeeds[i]);
    }
    fclose(out);

    cudaFree(devStructureSeeds);
    cudaFree(devWorldSeeds);
    free(hostStructureSeeds);
    free(hostWorldSeeds);
    printf("Finished writing %zu world seeds to %s\n", totalWorldSeeds, outputFile);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s river INPUTFILE OUTPUTFILE", argv[0]);
        fprintf(stderr, "  %s half INPUTFILE OUTPUTFILE", argv[0]);
        fprintf(stderr, "  %s structure INPUTFILE OUTPUTFILE", argv[0]);
        return 1;
    }

    const char* mode = argv[1];
    const char* inputFile = argv[2];
    const char* outputFile = argv[3];

    if (strcmp(mode, "river") == 0) {
        riverToHalfSeeds(inputFile, outputFile);
    } else if (strcmp(mode, "half") == 0) {
        halfToStructureSeeds(inputFile, outputFile);
    } else if (strcmp(mode, "structure") == 0) {
        structureToWorldSeedsGPU(inputFile, outputFile);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    return 0;
}
