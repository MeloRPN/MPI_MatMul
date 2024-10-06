from mpi4py import MPI
import numpy as np
### Multiplicação de Matrizes em Python com Computação Distribuída usando OpenMPI ###
#verificar requisitos !

#inicialização do MPI - mpi exec -n 4 python MPI_entrega.py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#função para dividir a matriz em blocos
def divide_matriz(matriz, num_blocos_linhas, num_blocos_colunas):
    blocos = []
    m, n = matriz.shape
    linhas_por_bloco = m // num_blocos_linhas
    colunas_por_bloco = n // num_blocos_colunas
    for i in range(num_blocos_linhas):
        linha_inicial = i * linhas_por_bloco
        linha_final = (i + 1) * linhas_por_bloco if i != num_blocos_linhas - 1 else m
        for j in range(num_blocos_colunas):
            coluna_inicial = j * colunas_por_bloco
            coluna_final = (j + 1) * colunas_por_bloco if j != num_blocos_colunas - 1 else n
            blocos.append(matriz[linha_inicial:linha_final, coluna_inicial:coluna_final])
    return blocos

#### definir as dimensões das matrizes ####
if rank == 0:
    m, n = 3000, 30  #mat A (m x n)
    n, p = 30, 3000 #mat B (n x p)
else:
    m, n, p = None, None, None

#broadcast das dimensões
m = comm.bcast(m, root=0)
n = comm.bcast(n, root=0)
p = comm.bcast(p, root=0)

#número de blocos em linhas e colunas
num_blocos = int(np.sqrt(size))  # Número de blocos por dimensão

#dimensões de blocos locais
linhas_por_bloco_A = m // num_blocos
colunas_por_bloco_A = n // num_blocos
linhas_por_bloco_B = n // num_blocos
colunas_por_bloco_B = p // num_blocos

#inicialização das matrizes A e B no processo root
if rank == 0:
    A = np.random.rand(m, n)
    B = np.random.rand(n, p)
    #divisão de blocos
    blocos_A = divide_matriz(A, num_blocos, num_blocos)
    blocos_B = divide_matriz(B, num_blocos, num_blocos)
else:
    A = None
    B = None
    blocos_A = None
    blocos_B = None

#scatterv precisa de tamanho_bloco e desloc para blocos
def get_counts_desloc(matrix_shape, bloco_F, num_processes):
    bloco_T = bloco_F[0] * bloco_F[1]  # Tamanho do bloco em elementos
    tamanho_bloco = np.full(num_processes, bloco_T, dtype=int)  # Cada processo envia blocos do mesmo tamanho
    desloc = np.zeros(num_processes, dtype=int)
    for i in range(1, num_processes):
        desloc[i] = desloc[i - 1] + tamanho_bloco[i - 1]  # Definir os deslocamentos
    return tamanho_bloco, desloc

# Para a matriz A e B, os tamanhos de blocos são iguais em cada processo
tamanho_bloco_A, desloc_A = get_counts_desloc((m, n), (linhas_por_bloco_A, colunas_por_bloco_A), size)
tamanho_bloco_B, desloc_B = get_counts_desloc((n, p), (linhas_por_bloco_B, colunas_por_bloco_B), size)

# Cada processo recebe um bloco local de A e B
bloco_A_local = np.empty((linhas_por_bloco_A, colunas_por_bloco_A), dtype='d')
bloco_B_local = np.empty((linhas_por_bloco_B, colunas_por_bloco_B), dtype='d')

# Scatterv para distribuir blocos da matriz A
comm.Scatterv([A if rank == 0 else None, tamanho_bloco_A, desloc_A, MPI.DOUBLE], bloco_A_local, root=0)

# Scatterv para distribuir blocos da matriz B
comm.Scatterv([B if rank == 0 else None, tamanho_bloco_B, desloc_B, MPI.DOUBLE], bloco_B_local, root=0)


# Alocação de matriz para armazenar o bloco de C que será calculado
bloco_C_local = np.zeros((bloco_A_local.shape[0], bloco_B_local.shape[1]))

# Multiplicação local do bloco A e B para calcular o bloco parcial de C
bloco_C_local += np.dot(bloco_A_local, bloco_B_local)

# Comunicação assíncrona para troca de blocos de A e B entre os processos
req_send_A = []
req_recv_A = []
req_send_B = []
req_recv_B = []

for i in range(num_blocos - 1):
    # Enviar o bloco de A para o próximo processo
    destino_A = (rank + 1) % size
    origem_A = (rank - 1 + size) % size
    req_send_A.append(comm.Isend(bloco_A_local, dest=destino_A))
    req_recv_A.append(comm.Irecv(bloco_A_local, source=origem_A))

    # Enviar o bloco de B para o próximo processo
    destino_B = (rank + 1) % size
    origem_B = (rank - 1 + size) % size
    req_send_B.append(comm.Isend(bloco_B_local, dest=destino_B))
    req_recv_B.append(comm.Irecv(bloco_B_local, source=origem_B))

    # Aguarda a troca assíncrona terminar
    MPI.Request.Waitall(req_recv_A)
    MPI.Request.Waitall(req_recv_B)

    # Multiplica os novos blocos de A e B
    bloco_C_local += np.dot(bloco_A_local, bloco_B_local)

# Recolher o resultado final na matriz C
C = None
if rank == 0:
    C = np.empty((m, p), dtype='d')

# Gatherv para reunir os blocos de C calculados por cada processo
tamanho_bloco_C, desloc_C = get_counts_desloc((m, p), (bloco_C_local.shape[0], bloco_C_local.shape[1]), size)
comm.Gatherv(bloco_C_local, [C if rank == 0 else None, tamanho_bloco_C, desloc_C, MPI.DOUBLE], root=0)

# Imprimir o resultado
if rank == 0:
    print("Matriz A (m x n):")
    print(A)
    print("\nMatriz B (n x p):")
    print(B)
    print("\nResultado da multiplicação A * B (m x p):")
    print(C)