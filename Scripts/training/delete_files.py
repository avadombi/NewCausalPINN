from Scripts.dependencies.dependencies import os, shutil


def delete_files_in_dict(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def delete_files(path_save):
    for j in range(1, 11):
        delete_files_in_dict(f'{path_save}/{j}')
