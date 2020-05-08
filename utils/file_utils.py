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
    if os.path.exists(path):
        if action is "override":
            os.makedirs(path)
        elif action is "duplicate":
            created_path = duplicate_path(path)
        elif action is "error":
            raise AssertionError("\'%s\' already exists." % path)
        else: 
            raise AssertionError("Invalid action is used.")
    else:
        os.makedirs(path)

    if verbose:
        print("\'%s\' is created." % (created_path))

    return created_path
