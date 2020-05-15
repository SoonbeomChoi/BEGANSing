import os 

def duplicate_path(path):
    duplicated_path = path
    n = 0
    while os.path.exists(duplicated_path):
        n += 1
        duplicated_path = path + '_' + str(n)
    
    os.makedirs(duplicated_path)

    return duplicated_path

def create_path(path, action="duplicate", verbose=True):
    created_path = path
    verbose_message = 'created'
    if os.path.exists(path):
        if action == 'overwrite':
            os.makedirs(path, exist_ok=True)
            verbose_message = 'overwritten'
        elif action == 'duplicate':
            created_path = duplicate_path(path)
        elif action == 'error':
            raise AssertionError("\'%s\' already exists." % path)
        else: 
            raise AssertionError("Invalid action is used.")
    else:
        os.makedirs(path)

    if verbose:
        print("\'%s\' is %s." % (created_path, verbose_message))

    return created_path
