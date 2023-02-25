import os
from settings import BASE_PATH


# Delete all the files and folders under the given path
def del_files(path):
    files = os.listdir(path)
    for file_name in files:
        file_path = os.path.join(path, file_name)
        os.remove(file_path)


if __name__ == '__main__':
    # Delete all the files under 'test-output/adjusted' and 'test-output/best ind'
    folder1_path = os.path.abspath(os.path.join(BASE_PATH, "test-output/adjusted/"))
    folder2_path = os.path.abspath(os.path.join(BASE_PATH, "test-output/best ind/"))
    del_files(folder1_path)
    del_files(folder2_path)

    file_path1 = os.path.abspath(os.path.join(BASE_PATH, "log_generations.txt"))
    file_path2 = os.path.abspath(os.path.join(BASE_PATH, "NGEN hof.gif"))
    if os.path.exists(file_path1):
        os.remove(file_path1)
    if os.path.exists(file_path2):
        os.remove(file_path2)


