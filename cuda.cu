#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define THREADS_PER_BLOCK 256
#define EXPANSION_SIZE 65536  // 2^16

// halftostructrue
__global__ void halfToStructureKernel(
    uint64_t* halfSeeds, int numHalfSeeds, uint64_t* outStructureSeeds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputSeeds = numHalfSeeds * EXPANSION_SIZE;
    if (idx >= totalOutputSeeds) return;

    int halfSeedIdx = idx / EXPANSION_SIZE;
    int upperBits = idx % EXPANSION_SIZE;

    uint64_t halfSeed = halfSeeds[halfSeedIdx];
    uint64_t structureSeed = ((uint64_t)upperBits << 32) | (halfSeed & 0xFFFFFFFFULL);

    outStructureSeeds[idx] = structureSeed;
}

// structuretononrandom
__global__ void structureToWorldKernel(
    uint64_t* structureSeeds, int numStructureSeeds, int64_t* outWorldSeeds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputSeeds = numStructureSeeds * EXPANSION_SIZE;
    if (idx >= totalOutputSeeds) return;

    int structSeedIdx = idx / EXPANSION_SIZE;
    int upperBits = idx % EXPANSION_SIZE;

    uint64_t structureSeed = structureSeeds[structSeedIdx];
    uint64_t lower48 = structureSeed & 0xFFFFFFFFFFFFULL;

    uint64_t combined = ((uint64_t)upperBits << 48) | lower48;

    int64_t worldSeed = (int64_t)(combined + 0x8000000000000000ULL) - 0x8000000000000000LL;

    outWorldSeeds[idx] = worldSeed;
}

// reads seeds
// free seeds
uint64_t* readSeedsFromFile(const char* filename, int* outCount) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    int capacity = 1024;
    uint64_t* seeds = (uint64_t*)malloc(capacity * sizeof(uint64_t));
    int count = 0;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (count >= capacity) {
            capacity *= 2;
            uint64_t* newSeeds = (uint64_t*)realloc(seeds, capacity * sizeof(uint64_t));
            if (!newSeeds) {
                fprintf(stderr, "Memory allocation error\n");
                free(seeds);
                fclose(f);
                return NULL;
            }
            seeds = newSeeds;
        }
        long long int seed = 0;
        if (sscanf(line, "%lld", &seed) == 1) {
            seeds[count++] = (uint64_t)seed;
        }
    }

    fclose(f);
    *outCount = count;
    return seeds;
}

// write to file
void writeSeedsToFileInt64(const char* filename, int64_t* seeds, size_t count) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening output file %s\n", filename);
        return;
    }
    for (size_t i = 0; i < count; i++) {
        fprintf(f, "%lld\n", (long long)seeds[i]);
    }
    fclose(f);
}

// write to file.
void writeSeedsToFileUint64(const char* filename, uint64_t* seeds, size_t count) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening output file %s\n", filename);
        return;
    }
    for (size_t i = 0; i < count; i++) {
        fprintf(f, "%llu\n", (unsigned long long)seeds[i]);
    }
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s half|structure [input_file] [output_file]\n", argv[0]);
        return 1;
    }

    const char* mode = argv[1];
    const char* inputFile = NULL;
    const char* outputFile = NULL;

    if (strcmp(mode, "half") == 0) {
        inputFile = (argc >= 3) ? argv[2] : "half_seeds.txt";
        outputFile = (argc >= 4) ? argv[3] : "structure_seeds.txt";

        int numHalfSeeds = 0;
        uint64_t* halfSeeds = readSeedsFromFile(inputFile, &numHalfSeeds);
        if (!halfSeeds) return 1;

        printf("Read %d half seeds from %s\n", numHalfSeeds, inputFile);

        size_t totalOutputSeeds = (size_t)numHalfSeeds * EXPANSION_SIZE;

        uint64_t* structureSeedsHost = (uint64_t*)malloc(totalOutputSeeds * sizeof(uint64_t));
        if (!structureSeedsHost) {
            fprintf(stderr, "Host memory allocation failed\n");
            free(halfSeeds);
            return 1;
        }

        uint64_t* halfSeedsDevice;
        uint64_t* structureSeedsDevice;

        cudaMalloc(&halfSeedsDevice, numHalfSeeds * sizeof(uint64_t));
        cudaMalloc(&structureSeedsDevice, totalOutputSeeds * sizeof(uint64_t));

        cudaMemcpy(halfSeedsDevice, halfSeeds, numHalfSeeds * sizeof(uint64_t), cudaMemcpyHostToDevice);

        int blocks = (totalOutputSeeds + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        printf("Launching kernel with %d blocks, %d threads per block\n", blocks, THREADS_PER_BLOCK);

        halfToStructureKernel<<<blocks, THREADS_PER_BLOCK>>>(halfSeedsDevice, numHalfSeeds, structureSeedsDevice);
        cudaDeviceSynchronize();

        cudaMemcpy(structureSeedsHost, structureSeedsDevice, totalOutputSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        writeSeedsToFileUint64(outputFile, structureSeedsHost, totalOutputSeeds);
        printf("Wrote %zu structure seeds to %s\n", totalOutputSeeds, outputFile);

        cudaFree(halfSeedsDevice);
        cudaFree(structureSeedsDevice);
        free(halfSeeds);
        free(structureSeedsHost);

    } else if (strcmp(mode, "structure") == 0) {
        inputFile = (argc >= 3) ? argv[2] : "structure_seeds.txt";
        outputFile = (argc >= 4) ? argv[3] : "world_seeds.txt";

        int numStructureSeeds = 0;
        uint64_t* structureSeeds = readSeedsFromFile(inputFile, &numStructureSeeds);
        if (!structureSeeds) return 1;

        printf("Read %d structure seeds from %s\n", numStructureSeeds, inputFile);

        size_t totalOutputSeeds = (size_t)numStructureSeeds * EXPANSION_SIZE;

        int64_t* worldSeedsHost = (int64_t*)malloc(totalOutputSeeds * sizeof(int64_t));
        if (!worldSeedsHost) {
            fprintf(stderr, "Failed\n");
            free(structureSeeds);
            return 1;
        }

        uint64_t* structureSeedsDevice;
        int64_t* worldSeedsDevice;

        cudaMalloc(&structureSeedsDevice, numStructureSeeds * sizeof(uint64_t));
        cudaMalloc(&worldSeedsDevice, totalOutputSeeds * sizeof(int64_t));

        cudaMemcpy(structureSeedsDevice, structureSeeds, numStructureSeeds * sizeof(uint64_t), cudaMemcpyHostToDevice);

        int blocks = (totalOutputSeeds + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        printf("Launching kernel with %d blocks, %d threads per block\n", blocks, THREADS_PER_BLOCK);

        structureToWorldKernel<<<blocks, THREADS_PER_BLOCK>>>(structureSeedsDevice, numStructureSeeds, worldSeedsDevice);
        cudaDeviceSynchronize();

        cudaMemcpy(worldSeedsHost, worldSeedsDevice, totalOutputSeeds * sizeof(int64_t), cudaMemcpyDeviceToHost);

        writeSeedsToFileInt64(outputFile, worldSeedsHost, totalOutputSeeds);
        printf("Wrote %zu world seeds to %s\n", totalOutputSeeds, outputFile);

        cudaFree(structureSeedsDevice);
        cudaFree(worldSeedsDevice);
        free(structureSeeds);
        free(worldSeedsHost);

    } else {
        fprintf(stderr, "Wrong mode '%s'. Use 'half' or 'structure'.\n", mode);
        return 1;
    }

    return 0;
}
