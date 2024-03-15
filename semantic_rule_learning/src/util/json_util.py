# taken from: https://git.fortiss.org/iiot_external/tsmatch/

def linearize(json_object):
    """
    Linearize a given json object, e.g. if it contains another json object inside:
    {object2: {key2: "value2}} -> {object2_key2: "value2"}
    or if it contains a list inside:
    {list: ["value1", "value2"]} -> {list_1: "value1", list_2: "value2"}
    This step makes processing the json data a lot easier for e.g. in the matchmaking process
    :param json_object:
    :return:
    """
    linearized_json = {}
    for attribute in json_object:
        # check if it is a dict or list
        if isinstance((json_object[attribute]), dict) or isinstance((json_object[attribute]), list):
            # extract the items from the list or dict
            if isinstance((json_object[attribute]), list):
                array_counter = 1
                for index in range(len(json_object[attribute])):
                    if isinstance(json_object[attribute][index], dict):
                        # call linearize function again for each of the list elements
                        linearized_inner_json = linearize(json_object[attribute][index])
                        for element2 in linearized_inner_json:
                            linearized_json[attribute + "_" + element2] = linearized_inner_json[element2]
                    else:
                        linearized_json[attribute + "_" + str(array_counter)] = json_object[attribute][index]
                        array_counter += 1
            else:
                # call linearize function again for each of the dict elements
                linearized_inner_json = linearize(json_object[attribute])
                for element in linearized_inner_json:
                    linearized_json[attribute + "_" + element] = linearized_inner_json[element]
        else:
            linearized_json[attribute] = json_object[attribute]

    return linearized_json
