def formatter(data):
    try:
        number = int(data)
        return number
    except ValueError:
        raise "String contains non-numeric characters."