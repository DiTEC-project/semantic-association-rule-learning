def snake_case_to_camel_case(string_in_snake_case):
    temp = string_in_snake_case.split("_")
    return temp[0] + ''.join(ele.title() for ele in temp[1:])