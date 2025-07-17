# HalftoStructuretoWorldNonRandom
Basically a Cuda Program to turn a list of half seeds into structure seeds or turn a list of structure seeds into non-random world seeds.

# Compile on Colab
Compile on Colab using this
```
!nvcc -o abc cuda.cu
```
Then run using this
```
!./abc half PATHTOHALFSEEDS
```
Or if your doing structure to non-random
```
!./abc structure PATHOFSTRUCTURESEEDS
```
Or if your doing river to half
```
!./abc structure PATHOFRIVERSEEDS
```

# Note
Also if you copy the file path
```
./content/src/structure_seeds.txt
```

Remove the content part
```
./src/structure_seeds.txt
```
