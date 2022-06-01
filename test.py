# from multiprocessing import Pool

# def f(x):
#     n = 0
#     for i in range(x):
#         n += i
#     return n

# with Pool(1000) as p:
#     print(p.map(f, [x for x in range(10000)]))

# print(list(map(f, [x for x in range(10000)])))

# import multiprocessing
# print(multiprocessing.cpu_count())

from glob import glob
print(len(glob('rationlaes/FT3/batch_*')))