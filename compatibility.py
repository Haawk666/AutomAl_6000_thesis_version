import core


def convert(obj, old_version, version):

    # Return obj in a compatible state
    # Return None if not possible!

    if old_version == [0, 0, 0]:
        # Set new attributes
        print('Updating...')
        obj.starting_index = None
        fresh_obj = obj
        print('updated!')
    elif old_version == [0, 0, 1]:
        # Set new attributes
        fresh_obj = obj
    else:
        fresh_obj = None

    return fresh_obj

