def forzen_var_list(param_dict, prefixes):
    ret_dict = {}
    for name, value in param_dict.items():
        keep = True
        for prefix in prefixes:
            if name.startwiths(prefix):
                keep = False
                continue

        if keep:
            ret_dict[name] = value

    return ret_dict
