# HalftoStructuretoWorldNonRandom
Basically a Cuda Program to turn a list of half seeds into structure seeds or turn a list of structure seeds into non-random world seeds. (Also can turn a list of river seeds into halfseeds)

# How to use on Colab
Log in or create a Google Account to use [https://colab.research.google.com].

Click Runtime, and select T4 GPU.

Click the files button (looks like a folder)

And right-click the files area to upload or create a new file.

Then create a code cell (should look like a "+ Code" button)

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
!./abc river PATHTORIVERSEEDS PATHTOOUTPUT
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

# Info
One Half Seed = 65536 Structure Seeds

One Structure Seed = 65536 World Seeds

One River Seed = 64 Half Seeds

If you want to go from structure to random world seeds (which is 0-2 world seeds from one structure seed) look at elenaran's program: https://github.com/elenaran/structure_to_random

Thanks NelS for providing me with the original python scripts for turning a half seed into structure seeds and structure seeds into all possible world seeds. 

https://github.com/nel-s
