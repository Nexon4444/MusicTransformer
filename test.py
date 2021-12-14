# import numpy as np
# import utils
# import random
# files = list(utils.find_files_by_extensions("./midi_processed", ['.pickle']))
# file_dict = {
#     'train': files[:int(len(files) * 0.8)],
#     'eval': files[int(len(files) * 0.8): int(len(files) * 0.9)],
#     'test': files[int(len(files) * 0.9):],
# }
#
# batch_files1 = random.sample(file_dict["test"], k=1)
# batch_files2 = random.sample(file_dict["test"], k=1)
# batch_files3 = random.sample(file_dict["test"], k=1)
#
# print(batch_files1)
# print(batch_files3)
# print(batch_files2)