# HalftoStructuretoWorldNonRandom
Basically a Cuda Program to turn a list of half seeds into structure seeds or turn a list of structure seeds into non-random world seeds. (Also can turn a list of river seeds into halfseeds)

# Compile on Colab
Compile on Colab using this
```
!nvcc -o abc cuda.cu
```
Then run using this
```
!./abc half PATHTOHALFSEEDS PATHTOOUTPUT
```
Or if your doing structure to non-random
```
!./abc structure PATHTOSTRUCTURESEEDS PATHTOOUTPUT
```
Or if your doing river to half
```
!./abc structure PATHTORIVERSEEDS PATHTOOUTPUT
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

Sorry that you have to create your own output file. The file writing is not working and I'm too lazy to fix it.
