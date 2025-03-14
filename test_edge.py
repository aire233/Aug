filename = 'SW-620/SW-620_A.txt'
with open(filename, 'r') as f:
    with open('edges.txt', 'w') as f2:
        # edges = []
        while True:
            l1, l2 = f.readline(), f.readline()
            n11, n12 = l1.strip().split(', ')
            # n21, n22 = l2.strip().split(', ')
            # if n11 == n22 and n12 == n21:
            #     # print('Symmetric')
            #     pass
            # else:
            #     print('Asymmetric')
            f2.write(f'{n11} {n12}\n')
